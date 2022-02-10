'''
Comprises various functions needed for mesh processing.
For ex. mesh smoothing, face normal calculation, face centroid calculation, 
mesh splitting, mesh thickness computation,
connected component analysis of mesh.

Sample usage for thickness computation
    import mesh_processing as mp
    distance_inner, distance_outer = mp.get_thickness_mesh(itk_image, mesh_type=='FC')

'''

import numpy as np
from sklearn.cluster import KMeans
import itk
import vtk
from vtk.util import numpy_support as ns
from itkwidgets import view
from skimage import measure

# Helper Functions for Mesh Processing

# Get Centroid of all the cells
def get_cell_centroid(vtk_mesh):
    centroid_array = np.zeros([vtk_mesh.GetNumberOfCells(), 3])
    num_of_cells = vtk_mesh.GetNumberOfCells()

    points_array = np.zeros([num_of_cells, 3, 3])

    for i in range(num_of_cells):
        c = vtk_mesh.GetCell(i)
        p = c.GetPoints()
        p1 = np.array(p.GetData())
        points_array[i] = p1

    sum_array = np.sum(points_array, axis=1)
    centroid_array = sum_array/3.0
    return centroid_array
  
# Get Normal of all the cells
def get_cell_normals(vtk_mesh):
    # Get Normals for the cells of the mesh
    normals_filter = vtk.vtkPolyDataNormals()
    normals_filter.SetInputData(vtk_mesh)
    normals_filter.ComputePointNormalsOn()
    normals_filter.ComputeCellNormalsOn()
    normals_filter.SplittingOff()
    normals_filter.ConsistencyOn()
    normals_filter.AutoOrientNormalsOff()
    normals_filter.Update()
    
    output1 = normals_filter.GetOutput()
    d1 = np.array(output1.GetCellData().GetNormals())
    
    return d1

# Get VTK mesh using vertices and faces array
def get_vtk_mesh(verts, faces):
    cells = vtk.vtkCellArray()

    # Add Cells
    for i in range(0, faces.shape[0]):
        tri = vtk.vtkTriangle()
        current_cell_points = faces[i]
        
        for i in range(3):
            point_id = current_cell_points[i]
            tri.GetPointIds().SetId(i, point_id)    
        cells.InsertNextCell(tri)

    # Create a poly data object
    vtk_mesh = vtk.vtkPolyData()
    # Set the points and vertices we created as the geometry and topology of the polydata
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(ns.numpy_to_vtk(verts, deep=True))

    vtk_mesh.SetPoints(vtk_points)
    vtk_mesh.SetPolys(cells)
    vtk_mesh.Modified()

    connectivityFilter = vtk.vtkPolyDataConnectivityFilter()
    connectivityFilter.SetInputData(vtk_mesh)
    connectivityFilter.SetExtractionModeToAllRegions()
    connectivityFilter.Update()
        
    valid_meshes = []
    append_polydata = vtk.vtkAppendPolyData()
    
    # Filter out small regions
    filter_thresh = 3000

    for i in range(connectivityFilter.GetNumberOfExtractedRegions()):
        connectivityFilter.AddSpecifiedRegion(i)
        connectivityFilter.SetExtractionModeToSpecifiedRegions()
        connectivityFilter.Update()
        
        out1 = connectivityFilter.GetOutput()
        
        if out1.GetNumberOfCells() > filter_thresh:
          polydata = vtk.vtkPolyData()
          polydata.ShallowCopy(out1)
          append_polydata.AddInputData(polydata)
        
        connectivityFilter.DeleteSpecifiedRegion(i)
    
    append_polydata.Update()
    combined_mesh = append_polydata.GetOutput()

    return combined_mesh

# Create a Mesh by selecting relevant faces only
def get_vtk_sub_mesh(input_mesh, inner_face_list):
    points = vtk.vtkPoints()
    cells = vtk.vtkCellArray()

    # Add Cells
    counter = 0
    point_ids_hash = {}
    for i in inner_face_list:
        tri = vtk.vtkTriangle()
        current_cell_points = input_mesh.GetCell(i).GetPointIds()
        
        if current_cell_points.GetNumberOfIds() != 3:
            print('Ids are not three Some Issue ', current_cell_points.GetNumberOfIds())
        
        for i in range(3):
            point_id = current_cell_points.GetId(i)
            
            if point_id in point_ids_hash:
                new_point_id = point_ids_hash[point_id]
            else:
                new_point_id = counter
                point_ids_hash[point_id] = new_point_id
                counter = counter + 1
            
            tri.GetPointIds().SetId(i, new_point_id)
        
        cells.InsertNextCell(tri)

    points.SetNumberOfPoints(len(list(point_ids_hash.keys())))

    # Add points
    for i in point_ids_hash:
        p = input_mesh.GetPoint(i)
        new_id = point_ids_hash[i]
        points.SetPoint(new_id, p)

    # Create a poly data object
    polydata = vtk.vtkPolyData()
    # Set the points and vertices we created as the geometry and topology of the polydata
    polydata.SetPoints(points)
    polydata.SetPolys(cells)
    polydata.Modified()

    return polydata

# For splitting the Tibial cartilage
def split_tibial_cartilage_surface(mesh, mesh_normals, mesh_centroids):
    mesh_centroids_normalized = (mesh_centroids - np.mean(mesh_centroids, axis=0)) / \
                                (np.max(mesh_centroids, axis=0) - np.min(mesh_centroids, axis=0))

    # clustering normals
    features = np.concatenate((mesh_centroids_normalized * 1, mesh_normals * 10), axis=1)
    
    est = KMeans(n_clusters=2, algorithm="full")
    labels = est.fit(features).labels_

    # transfer 0/1 labels to -1/1 labels
    inner_outer_label_list = labels * 2 - 1

    # set inner surface which contains mostly positive normals
    if mesh_normals[inner_outer_label_list == -1, 1].mean() < 0:
        inner_outer_label_list = -inner_outer_label_list


    inner_face_list = np.where(inner_outer_label_list == -1)[0]
    outer_face_list = np.where(inner_outer_label_list == 1)[0]

    inner_mesh = get_vtk_sub_mesh(mesh, inner_face_list)
    outer_mesh = get_vtk_sub_mesh(mesh, outer_face_list)

    return inner_mesh, outer_mesh, inner_face_list, outer_face_list

# Clusters the feature vectors and segments the mesh
def cluster_and_segment(mesh_centroids_normalized, face_normal_value, dot_output):
    features = np.concatenate((mesh_centroids_normalized * 1, face_normal_value, dot_output), axis=1)
    est1 = KMeans(n_clusters=2, 
                 algorithm="full",
                 n_init=5)
    labels_upper = est1.fit(features).labels_
    labels_upper = labels_upper*2 - 1

    # set inner surface which contains mostly positive normals
    if face_normal_value[labels_upper == -1, 1].mean() < 0:
       labels_upper = -labels_upper
    
    return labels_upper
  
# For splitting the Femoral cartilage
def split_femoral_cartilage_surface(mesh, face_normal, face_centroid, num_divisions=3):
    mesh_centroids_normalized = (face_centroid - np.mean(face_centroid, axis=0)) / \
                                (np.max(face_centroid, axis=0) - np.min(face_centroid, axis=0))
    
    (xmin,xmax,ymin,ymax,zmin,zmax)  = mesh.GetBounds()
    bbox_min = np.array([xmin, ymin, zmin])
    bbox_max = np.array([xmax, ymax, zmax])

    center = (bbox_min + bbox_max) / 2
    
    inner_outer_label_list = np.zeros(mesh.GetNumberOfCells())  # up:1, down:-1
    connect_direction = center - face_centroid

    dot_output = np.multiply(connect_direction, face_normal)
    x_coord = mesh_centroids_normalized[:, 0]

    # For puting the labels at correct indices for all the segments
    inner_outer_label_list = np.zeros(mesh_centroids_normalized.shape[0])

    # For dividing the mesh into smaller segments for better clustering
    min_x = np.min(mesh_centroids_normalized[:, 0])
    max_x = np.max(mesh_centroids_normalized[:, 0])
    step_value = (max_x-min_x)/num_divisions

    # Perform clustering for each segment individually
    for i in range(num_divisions):
      lower_x = min_x + step_value*i 
      upper_x = lower_x + step_value
      current_indices = np.where((x_coord >= lower_x) & (x_coord < upper_x))[0]
      
      mesh_centroids_normalized_extracted = mesh_centroids_normalized[current_indices]
      face_normal_value_extracted = face_normal[current_indices]
      dot_output_extracted = dot_output[current_indices]

      # Clustering the current segment
      current_labels = cluster_and_segment(mesh_centroids_normalized_extracted, face_normal_value_extracted, dot_output_extracted)

      # Put the labels into the correct positions in the full mesh
      np.put(inner_outer_label_list, current_indices,  current_labels)
    
    inner_face_list = np.where(inner_outer_label_list == -1)[0]
    outer_face_list = np.where(inner_outer_label_list == 1)[0]

    inner_mesh = get_vtk_sub_mesh(mesh, inner_face_list)
    outer_mesh = get_vtk_sub_mesh(mesh, outer_face_list)

    return inner_mesh, outer_mesh, inner_face_list, outer_face_list
  
def smooth_mesh(input_mesh, num_iterations=150):
  smoothing_filter = vtk.vtkSmoothPolyDataFilter()
  smoothing_filter.SetNumberOfIterations(num_iterations)

  smoothing_filter.SetInputData(input_mesh)
  smoothing_filter.Update()

  output_mesh = smoothing_filter.GetOutput()
  return output_mesh

# For getting nearest neighbour distance between inner and outer mesh
def get_distance(inner_mesh, outer_mesh):
    distance_filter = vtk.vtkDistancePolyDataFilter()
    distance_filter.SetInputData(0, inner_mesh)
    distance_filter.SetInputData(1, outer_mesh)
    distance_filter.SignedDistanceOff()
    distance_filter.SetComputeSecondDistance(True)
    distance_filter.Update()

    distance_inner = distance_filter.GetOutput()
    distance_outer = distance_filter.GetSecondDistanceOutput()

    return distance_inner, distance_outer

def get_thickness_mesh(itk_image, mesh_type='FC'):
    '''
    Takes the probability map obtained from the segmentation algorithm as an itk image.
    Constructs a VTK mesh from it and returns the thickness between the inner and outer splitted mesh.
    Takes as argument the type of mesh 'FC' or 'TC'.
    '''
    spacing = itk_image.GetSpacing()
    img_array = np.swapaxes(np.asarray(itk_image), 0, 2).astype(float)
    
    # Obtain the mesh from Probability maps using Marching Cubes
    verts, faces, normals, values = measure.marching_cubes_lewiner(img_array, 0.5,
                                                                    spacing=spacing,
                                                                    step_size=1, 
                                                                    gradient_direction="ascent")
    
    mesh = get_vtk_mesh(verts, faces)

    # For smoothing the mesh surface to obtain gradually varying face normals
    mesh = smooth_mesh(mesh)
    
    # Obtain the cell normals and centroids to be used for splittig the cartilage
    mesh_cell_normals = get_cell_normals(mesh)
    mesh_cell_centroids = get_cell_centroid(mesh)

    # Split the mesh
    if mesh_type == 'FC':
        inner_mesh, outer_mesh, inner_face_list, outer_face_list = split_femoral_cartilage_surface(mesh,
                                                                                           mesh_cell_normals,
                                                                                           mesh_cell_centroids)
    else:
        inner_mesh, outer_mesh, inner_face_list, outer_face_list = split_tibial_cartilage_surface(mesh,
                                                                                           mesh_cell_normals,
                                                                                           mesh_cell_centroids)
    # Get the distance between inner and outer mesh
    distance_inner, distance_outer = get_distance(inner_mesh, outer_mesh)

    return distance_inner, distance_outer