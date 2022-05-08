# All Imports

import numpy as np

import os
os.environ["CUDA_VISIBLE_DEVICES"]=""


import coiled
import dask
from dask import delayed, compute, visualize
from dask.distributed import Client, progress, LocalCluster



@delayed(nout=3)
def register_images_delayed():
    import boto3
    import itk
    import icon_registration
    import icon_registration.itk_wrapper as itk_wrapper
    import icon_registration.pretrained_models as pretrained_models
    from os.path import exists
    import torch
    torch.set_num_threads(8)
    
    image_A = 'image_preprocessed.nii.gz'
    image_B = 'atlas_image.nii.gz'
    if (exists(image_A) and exists(image_B)) == False:
        s3          = boto3.resource("s3")
        bucket_name = 'oaisample1'
        bucket      = s3.Bucket(bucket_name)

        s3.Bucket(bucket_name).download_file(image_A, image_A)
        s3.Bucket(bucket_name).download_file(image_B, image_B)
    
    image_A = itk.imread(image_A, itk.D)
    image_B = itk.imread(image_B, itk.D)

    model = pretrained_models.OAI_knees_gradICON_model()
    model.to('cpu')

    # Register the images
    phi_AB, phi_BA = itk_wrapper.register_pair(model, image_A, image_B)
    return itk.dict_from_transform(phi_AB), itk.dict_from_image(image_A), itk.dict_from_image(image_B)


@delayed
def deform_probmap_delayed(phi_AB, image_A, image_B, image_type ='FC'):
    import itk
    import boto3

    s3          = boto3.resource("s3")
    bucket_name = 'oaisample1'
    bucket      = s3.Bucket(bucket_name)
    prob_file = str(image_type)+'_probmap.nii.gz'
    s3.Bucket(bucket_name).download_file(prob_file, prob_file)
    prob = itk.imread(prob_file, itk.D)

    phi_AB1  = itk.transform_from_dict(phi_AB)
    
    def set_parameters(phi_AB, phi_AB1):
        for i in range(len(phi_AB)-1):
            transform1 = phi_AB1.GetNthTransform(i)

            fp = phi_AB[i+1]['fixedParameters']
            o1 = transform1.GetFixedParameters()
            o1.SetSize(fp.shape[0])
            for j, v in enumerate(fp):
                o1.SetElement(j, v)
            transform1.SetFixedParameters(o1)

            p = phi_AB[i+1]['parameters']
            o2 = transform1.GetParameters()
            o2.SetSize(p.shape[0])
            for j, v in enumerate(p):
                o2.SetElement(j, v)
            transform1.SetParameters(o2)

    set_parameters(phi_AB, phi_AB1)
    image_A = itk.image_from_dict(image_A)
    image_B = itk.image_from_dict(image_B)

    interpolator = itk.LinearInterpolateImageFunction.New(image_A)
    
    warped_image = itk.resample_image_filter(prob, 
       transform=phi_AB1, 
       interpolator=interpolator,
       size=itk.size(image_B),
       output_spacing=itk.spacing(image_B),
       output_direction=image_B.GetDirection(),
       output_origin=image_B.GetOrigin()
    )

    output_dict = itk.dict_from_image(warped_image)
    return output_dict

@delayed(nout=1)
def get_thickness(warped_image, mesh_type):
    import itk
    import vtk
    import numpy as np
    from oai_analysis_2 import mesh_processing as mp

    def get_itk_mesh(vtk_mesh):
        Dimension = 3
        PixelType = itk.D
        
        # Get points array from VTK mesh
        points = vtk_mesh.GetPoints().GetData()
        points_numpy = np.array(points).flatten()#.astype('float32')
            
        polys = vtk_mesh.GetPolys().GetData()
        polys_numpy = np.array(polys).flatten()

        # Triangle Mesh
        vtk_cells_count = vtk_mesh.GetNumberOfPolys()
        polys_numpy = np.reshape(polys_numpy, [vtk_cells_count, Dimension+1])

        # Extracting only the points by removing first column that denotes the VTK cell type
        polys_numpy = polys_numpy[:, 1:]
        polys_numpy = polys_numpy.flatten().astype(np.uint64)

        # Get point data from VTK mesh to insert in ITK Mesh
        point_data_numpy = np.array(vtk_mesh.GetPointData().GetScalars())#.astype('float64')
        
        # Get cell data from VTK mesh to insert in ITK Mesh
        cell_data_numpy = np.array(vtk_mesh.GetCellData().GetScalars())#.astype('float64')
        
        MeshType = itk.Mesh[PixelType, Dimension]
        itk_mesh = MeshType.New()
        
        itk_mesh.SetPoints(itk.vector_container_from_array(points_numpy))
        itk_mesh.SetCellsArray(itk.vector_container_from_array(polys_numpy), itk.CommonEnums.CellGeometry_TRIANGLE_CELL)
        itk_mesh.SetPointData(itk.vector_container_from_array(point_data_numpy))
        itk_mesh.SetCellData(itk.vector_container_from_array(cell_data_numpy))    
        return itk_mesh

    warped_image = itk.image_from_dict(warped_image)
    distance_inner, distance_outer = mp.get_thickness_mesh(warped_image, mesh_type=mesh_type)
    distance_inner_itk = get_itk_mesh(distance_inner)
    distance_inner_itk_dict = itk.dict_from_mesh(distance_inner_itk)

    return distance_inner_itk_dict

@delayed(nout=2)
def segment_image_delayed():
    from os.path import exists
    import boto3
    import itk
    from oai_analysis_2 import analysis_object as ao
    import torch
    torch.set_num_threads(8)

    image_A = 'image_preprocessed.nii.gz'
    if exists(image_A) ==  False:
        s3          = boto3.resource("s3")
        bucket_name = 'oaisample1'
        bucket      = s3.Bucket(bucket_name)
        s3.Bucket(bucket_name).download_file(image_A, image_A)

    test_volume = itk.imread(image_A)
    obj = ao.AnalysisObject()
    FC_prob, TC_prob = obj.segment(test_volume)

    return itk.dict_from_image(FC_prob), itk.dict_from_image(TC_prob)