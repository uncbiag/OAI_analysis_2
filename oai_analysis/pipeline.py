import os
import pathlib

import icon_registration.itk_wrapper as itk_wrapper
import itk
import numpy as np
import vtk
from unigradicon import preprocess, get_unigradicon
from vtk import vtkPointLocator
from vtk.util.numpy_support import numpy_to_vtk

import mesh_processing as mp
from analysis_object import AnalysisObject
from cartilage_shape_processing import thickness_3d_to_2d
from thickness_computation import compute_thickness

DATA_DIR = pathlib.Path(__file__).parent / "data"


def write_vtk_mesh(mesh, filename):
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(mesh)
    writer.SetFileVersion(42)  # ITK does not support newer version (5.1)
    writer.SetFileTypeToBinary()  # reading and writing binary files is faster
    writer.Write()


def transform_mesh(mesh, transform, filename_prefix, keep_intermediate_outputs):
    """
    Transform the input mesh using the provided transform.

    :param mesh: input mesh
    :param transform: The modelling transform to use (the inverse of image resampling transform)
    :param filename_prefix: prefix (including path) for the intermediate file names
    :return: transformed mesh
    """
    itk_mesh = mp.get_itk_mesh(mesh)
    t_mesh = itk.transform_mesh_filter(itk_mesh, transform=transform)
    # itk.meshwrite(t_mesh, filename_prefix + "_transformed.vtk", binary=True)  # does not work in 5.4 and earlier
    itk.meshwrite(t_mesh, filename_prefix + "_transformed.vtk", compression=True)

    transformed_mesh = mp.read_vtk_mesh(filename_prefix + "_transformed.vtk")

    if keep_intermediate_outputs:
        write_vtk_mesh(mesh, filename_prefix + "_original.vtk")
    else:
        os.remove(filename_prefix + "_transformed.vtk")

    return transformed_mesh


def into_canonical_orientation(image):
    """
    Reorient the given image into the canonical orientation.

    :param image: input image
    :return: reoriented image
    """
    dicom_lps = itk.SpatialOrientationEnums.ValidCoordinateOrientations_ITK_COORDINATE_ORIENTATION_RAI
    dicom_ras = itk.SpatialOrientationEnums.ValidCoordinateOrientations_ITK_COORDINATE_ORIENTATION_LPI
    dicom_pir = itk.SpatialOrientationEnums.ValidCoordinateOrientations_ITK_COORDINATE_ORIENTATION_ASL
    oriented_image = itk.orient_image_filter(
        image,
        use_image_direction=True,
        # given_coordinate_orientation=dicom_lps,
        # desired_coordinate_orientation=dicom_ras,
        desired_coordinate_orientation=dicom_pir,  # atlas' orientation
    )
    return oriented_image


def project_distance_to_atlas(thickness_image, atlas_mesh):
    num_points = atlas_mesh.GetNumberOfPoints()
    thickness = np.zeros(num_points, dtype=np.float32)
    for i in range(num_points):
        point = atlas_mesh.GetPoint(i)
        image_index = thickness_image.TransformPhysicalPointToIndex(point)
        thickness[i] = thickness_image.GetPixel(image_index)

    vtk_array = numpy_to_vtk(thickness, deep=True, array_type=vtk.VTK_FLOAT)
    vtk_array.SetName("vertex_thickness")
    atlas_mesh.GetPointData().AddArray(vtk_array)
    return atlas_mesh


def analysis_pipeline(input_path, output_path, keep_intermediate_outputs):
    """
    Computes cartilage thickness for femur and tibia from knee MRI.

    :param input_path: path to the input image file, or path to input directory containing DICOM image series.
    :param output_path: path to the desired directory for outputs.
    """
    in_image = itk.imread(input_path, pixel_type=itk.F)
    in_image = into_canonical_orientation(in_image)  # simplifies mesh processing
    in_image = preprocess(in_image, modality="mri")
    os.makedirs(output_path, exist_ok=True)  # also holds intermediate results
    if keep_intermediate_outputs:
        itk.imwrite(in_image, os.path.join(output_path, "in_image.nrrd"))

    print("Segmenting the cartilage")
    obj = AnalysisObject()
    FC_prob, TC_prob = obj.segment(in_image)
    if keep_intermediate_outputs:
        itk.imwrite(FC_prob, os.path.join(output_path, "FC_prob.nrrd"))
        itk.imwrite(TC_prob, os.path.join(output_path, "TC_prob.nrrd"))

    fc_thickness_image, fc_distance = compute_thickness(FC_prob)
    tc_thickness_image, tc_distance = compute_thickness(TC_prob)
    if keep_intermediate_outputs:
        itk.imwrite(fc_thickness_image, os.path.join(output_path, "fc_thickness_image.nrrd"), compression=True)
        itk.imwrite(tc_thickness_image, os.path.join(output_path, "tc_thickness_image.nrrd"), compression=True)
        itk.imwrite(fc_distance, os.path.join(output_path, "fc_distance.nrrd"), compression=True)
        itk.imwrite(tc_distance, os.path.join(output_path, "tc_distance.nrrd"), compression=True)

    atlas_filename = DATA_DIR / "atlases/atlas_60_LEFT_baseline_NMI/atlas.nii.gz"
    atlas_image = itk.imread(atlas_filename, itk.F)

    inner_mesh_fc_atlas = mp.read_vtk_mesh(
        DATA_DIR / "atlases/atlas_60_LEFT_baseline_NMI/atlas_FC_inner_mesh_smooth.ply")
    inner_mesh_tc_atlas = mp.read_vtk_mesh(
        DATA_DIR / "atlases/atlas_60_LEFT_baseline_NMI/atlas_TC_inner_mesh_smooth.ply")

    print("Registering the input image to the atlas")
    model = get_unigradicon()
    in_image_D = in_image.astype(itk.D)
    atlas_image_D = atlas_image.astype(itk.D)
    phi_AB, phi_BA = itk_wrapper.register_pair(model, in_image_D, atlas_image_D, finetune_steps=None)
    if keep_intermediate_outputs:
        print("Saving registration results")
        itk.transformwrite(phi_AB, os.path.join(output_path, "resampling.tfm"))
        itk.transformwrite(phi_BA, os.path.join(output_path, "modelling.tfm"))

    print("Transforming the thickness measurements to the atlas space")
    transformed_fc_thickness = itk.resample_image_filter(
        fc_thickness_image, transform=phi_AB, reference_image=atlas_image, use_reference_image=True)
    transformed_tc_thickness = itk.resample_image_filter(
        tc_thickness_image, transform=phi_AB, reference_image=atlas_image, use_reference_image=True)
    if keep_intermediate_outputs:
        itk.imwrite(transformed_fc_thickness,
                    os.path.join(output_path, "transformed_fc_thickness.nrrd"), compression=True)
        itk.imwrite(transformed_tc_thickness,
                    os.path.join(output_path, "transformed_tc_thickness.nrrd"), compression=True)

    print("Mapping the thickness to the atlas mesh")
    mapped_mesh_fc = project_distance_to_atlas(transformed_fc_thickness, inner_mesh_fc_atlas)
    mapped_mesh_tc = project_distance_to_atlas(transformed_tc_thickness, inner_mesh_tc_atlas)
    if keep_intermediate_outputs:
        write_vtk_mesh(mapped_mesh_fc, output_path + "/mapped_mesh_fc.vtk")
        write_vtk_mesh(mapped_mesh_tc, output_path + "/mapped_mesh_tc.vtk")

    print("Projecting thickness to 2D")
    thickness_3d_to_2d(mapped_mesh_fc, mesh_type='FC', output_filename=output_path + '/thickness_FC')
    thickness_3d_to_2d(mapped_mesh_tc, mesh_type='TC', output_filename=output_path + '/thickness_TC')


if __name__ == "__main__":
    test_cases = [
        r"M:\Dev\Osteoarthritis\OAI_analysis_2\oai_analysis\data\test_data\colab_case\image_preprocessed.nii.gz",
        r"M:\Dev\Osteoarthritis\NAS\OAIBaselineImages\results\0.C.2\9000798\20040924\10249506",
        r"M:\Dev\Osteoarthritis\NAS\OAIBaselineImages\results\0.C.2\9000798\20040924\10249512",
        r"M:\Dev\Osteoarthritis\NAS\OAIBaselineImages\results\0.C.2\9000798\20040924\10249516",
        # r"M:\Dev\Osteoarthritis\NAS\OAIBaselineImages\results\0.C.2\9000798\20040924\10249517",  # parsed incorrectly
    ]

    for i, case in enumerate(test_cases):
        print(f"\nProcessing case {case}")
        output_path = f"./OAI_results/case_{i:03d}"
        analysis_pipeline(case, output_path, keep_intermediate_outputs=True)
