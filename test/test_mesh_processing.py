import pytest

from oai_analysis import mesh_processing as mp
import trimesh
from pathlib import Path
import numpy as np
import itk

vtk = pytest.importorskip("vtk")

def test_get_mesh():
    test_prob_filepath = Path(__file__).parent / "test_files" / "colab_case" / "TC_probmap.nii.gz"
    image = itk.imread(test_prob_filepath)
    mesh = mp.get_mesh_from_probability_map(image)

    baseline_mesh_filepath = Path(__file__).parent / "test_files" / "colab_case" / "TC_mesh.vtk"
    baseline = itk.meshread(baseline_mesh_filepath)

    trimesh = mp.get_trimesh(mesh)
    baseline_trimesh = mp.get_trimesh(baseline)
    np.testing.assert_allclose(trimesh.vertices, baseline_trimesh.vertices, atol=0.02)


def test_get_cell_normals():
    test_filepath = Path(__file__).parent / "test_files" / "colab_case" / "avsm" / "TC_mesh_world.ply"

    # test_filepath = Path(__file__).parent / "test_files" / "colab_case" / "avsm" / "TC_mesh_world.off"
    # mesh = itk.meshread(str(test_filepath))
    # mesh = mp.get_trimesh(mesh)

    mesh = trimesh.load_mesh(str(test_filepath))

    ply_reader = vtk.vtkPLYReader()
    ply_reader.SetFileName(str(test_filepath))
    ply_reader.Update()
    polydata = ply_reader.GetOutput()

    # Get Normal of all the cells
    def vtk_get_cell_normals(vtk_mesh):
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

    vtk_cell_normals = vtk_get_cell_normals(polydata)

    np.testing.assert_allclose(mesh.face_normals, vtk_cell_normals, atol=0.02)


def test_get_cell_centroid():
    test_filepath = Path(__file__).parent / "test_files" / "colab_case" / "avsm" / "TC_mesh_world.ply"

    mesh = trimesh.load_mesh(str(test_filepath))
    def trimesh_get_cell_centroid(mesh):
        centroid_array = np.zeros([len(mesh.faces), 3])
        num_of_cells = len(mesh.faces)

        points_array = np.zeros([num_of_cells, 3, 3])

        for i in range(num_of_cells):
            c = mesh.faces[i]
            p = [mesh.vertices[c[0]], mesh.vertices[c[1]], mesh.vertices[c[2]]]
            points_array[i] = p

        sum_array = np.sum(points_array, axis=1)
        centroid_array = sum_array / 3.0
        return centroid_array

    def vtk_get_cell_centroid(vtk_mesh):
        centroid_array = np.zeros([vtk_mesh.GetNumberOfCells(), 3])
        num_of_cells = vtk_mesh.GetNumberOfCells()

        points_array = np.zeros([num_of_cells, 3, 3])

        for i in range(num_of_cells):
            c = vtk_mesh.GetCell(i)
            p = c.GetPoints()
            p1 = np.array(p.GetData())
            points_array[i] = p1

        sum_array = np.sum(points_array, axis=1)
        centroid_array = sum_array / 3.0
        return centroid_array

    ply_reader = vtk.vtkPLYReader()
    ply_reader.SetFileName(str(test_filepath))
    ply_reader.Update()
    polydata = ply_reader.GetOutput()

    vtk_centroid = vtk_get_cell_centroid(polydata)

    trimesh_centroid = trimesh_get_cell_centroid(mesh)

    np.testing.assert_allclose(vtk_centroid, trimesh_centroid, atol=0.02)