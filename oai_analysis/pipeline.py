import os
import pathlib
import sys

import icon_registration.itk_wrapper as itk_wrapper
import itk
import numpy as np
import vtk
from unigradicon import get_unigradicon
from vtk.util.numpy_support import numpy_to_vtk

import mesh_processing as mp
from analysis_object import AnalysisObject
from cartilage_shape_processing import thickness_3d_to_2d
from thickness_computation import compute_thickness

DATA_DIR = pathlib.Path(__file__).parent / "data"



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
    transformed_mesh.GetPointData().AddArray(mesh.GetPointData().GetArray(0))  # transfer thickness

    if keep_intermediate_outputs:
        mp.write_vtk_mesh(mesh, filename_prefix + "_original.vtk")
    else:
        os.remove(filename_prefix + "_transformed.vtk")

    return transformed_mesh


def preprocess(image, window_min_percentile=0.1, window_max_percentile=99.9, output_min=0.0, output_max=1.0):
    image_array = itk.GetArrayViewFromImage(image)
    win_min = float(np.percentile(image_array, window_min_percentile))
    win_max = float(np.percentile(image_array, window_max_percentile))
    result = itk.intensity_windowing_image_filter(image,
                                                  window_minimum=win_min,
                                                  window_maximum=win_max,
                                                  output_minimum=output_min,
                                                  output_maximum=output_max)
    return result


def into_canonical_orientation(image, flip_left_right):
    """
    Reorient the given image into the canonical orientation.

    :param image: input image
    :param flip_left_right: if True, flips the image left-right
    :return: reoriented image
    """
    dicom_lps = itk.SpatialOrientationEnums.ValidCoordinateOrientations_ITK_COORDINATE_ORIENTATION_RAI
    dicom_ras = itk.SpatialOrientationEnums.ValidCoordinateOrientations_ITK_COORDINATE_ORIENTATION_LPI
    dicom_pir = itk.SpatialOrientationEnums.ValidCoordinateOrientations_ITK_COORDINATE_ORIENTATION_ASL
    if flip_left_right:
        image_dir = itk.array_from_matrix(image.GetDirection())
        image_dir[0, :] *= -1  # flip left-right
        image.SetDirection(itk.matrix_from_array(image_dir))
    oriented_image = itk.orient_image_filter(
        image,
        use_image_direction=True,
        # given_coordinate_orientation=dicom_lps,
        # desired_coordinate_orientation=dicom_ras,
        desired_coordinate_orientation=dicom_pir,  # atlas' orientation
    )
    return oriented_image


def sample_distance_from_image(thickness_image, mesh):
    num_points = mesh.GetNumberOfPoints()
    thickness = np.zeros(num_points, dtype=np.float32)
    for i in range(num_points):
        point = mesh.GetPoint(i)
        image_index = thickness_image.TransformPhysicalPointToIndex(point)
        thickness[i] = thickness_image.GetPixel(image_index)

    vtk_array = numpy_to_vtk(thickness, deep=True, array_type=vtk.VTK_FLOAT)
    vtk_array.SetName("vertex_thickness")
    mesh.GetPointData().AddArray(vtk_array)
    return mesh


def analysis_pipeline(input_path, output_path, laterality, keep_intermediate_outputs):
    """
    Computes cartilage thickness for femur and tibia from knee MRI.

    :param input_path: path to the input image file, or path to input directory containing DICOM image series.
    :param output_path: path to the desired directory for outputs.
    """
    in_image = itk.imread(input_path, pixel_type=itk.F)
    if pathlib.Path(input_path).is_dir():  # DICOM series
        namesGenerator = itk.GDCMSeriesFileNames.New()
        namesGenerator.SetUseSeriesDetails(True)
        namesGenerator.SetDirectory(input_path)
        first_slice = itk.imread(namesGenerator.GetInputFileNames()[0])
        metadata = first_slice.GetMetaDataDictionary()
    else:
        metadata = in_image.GetMetaDataDictionary()
    if metadata.HasKey("0008|103e"):
        print(f"Laterality: {laterality}, Series description: {metadata['0008|103e']}")

    in_image = into_canonical_orientation(in_image, laterality == "right")  # simplifies mesh processing
    in_image = preprocess(in_image)
    os.makedirs(output_path, exist_ok=True)  # also holds intermediate results
    if keep_intermediate_outputs:
        itk.imwrite(in_image, os.path.join(output_path, "in_image.nrrd"))

    print("Segmenting the cartilage")
    obj = AnalysisObject()
    FC_prob, TC_prob = obj.segment(in_image)
    if keep_intermediate_outputs:
        itk.imwrite(FC_prob, os.path.join(output_path, "FC_prob.nrrd"))
        itk.imwrite(TC_prob, os.path.join(output_path, "TC_prob.nrrd"))

    atlas_filename = DATA_DIR / "atlases/atlas_60_LEFT_baseline_NMI/atlas_image.nii.gz"
    atlas_image = itk.imread(atlas_filename, itk.F)

    inner_mesh_fc_atlas = mp.read_vtk_mesh(
        DATA_DIR / "atlases/atlas_60_LEFT_baseline_NMI/atlas_FC_inner_mesh_LPS.ply")
    inner_mesh_tc_atlas = mp.read_vtk_mesh(
        DATA_DIR / "atlases/atlas_60_LEFT_baseline_NMI/atlas_TC_inner_mesh_LPS.ply")

    print("Registering the input image to the atlas")
    model = get_unigradicon()
    in_image_D = in_image.astype(itk.D)
    atlas_image_D = atlas_image.astype(itk.D)
    phi_AB, phi_BA = itk_wrapper.register_pair(model, in_image_D, atlas_image_D, finetune_steps=2)
    if keep_intermediate_outputs:
        print("Saving registration results")
        itk.transformwrite(phi_AB, os.path.join(output_path, "resampling.tfm"))
        itk.transformwrite(phi_BA, os.path.join(output_path, "modelling.tfm"))

    print("Converting segmentation probability images to meshes")
    # Get mesh from itk image
    fc_mesh_itk = mp.get_mesh_from_probability_map(FC_prob)
    tc_mesh_itk = mp.get_mesh_from_probability_map(TC_prob)
    fc_mesh = mp.itk_mesh_to_vtk_mesh(fc_mesh_itk)
    tc_mesh = mp.itk_mesh_to_vtk_mesh(tc_mesh_itk)
    if keep_intermediate_outputs:
        mp.write_vtk_mesh(fc_mesh, output_path + "/FC_mesh_patient.vtk")
        mp.write_vtk_mesh(tc_mesh, output_path + "/TC_mesh_patient.vtk")

    thickness_via_mesh_splitting = True
    if thickness_via_mesh_splitting:
        print("Computing the thickness map via mesh splitting into inner and outer")
        # Transform to atlas space for splitting
        fc_mesh_atlas = transform_mesh(fc_mesh, phi_BA, output_path + "/FC_mesh", False)
        tc_mesh_atlas = transform_mesh(tc_mesh, phi_BA, output_path + "/TC_mesh", False)
        fc_inner_atlas, fc_outer_atlas = mp.get_split_mesh(fc_mesh_atlas, inner_mesh_fc_atlas, mesh_type='FC')
        tc_inner_atlas, tc_outer_atlas = mp.get_split_mesh(tc_mesh_atlas, inner_mesh_tc_atlas, mesh_type='TC')

        # Transform back to patient space for distance measuring
        fc_inner_patient = transform_mesh(fc_inner_atlas, phi_AB, output_path + "/FC_mesh", False)
        fc_outer_patient = transform_mesh(fc_outer_atlas, phi_AB, output_path + "/FC_mesh", False)
        tc_inner_patient = transform_mesh(tc_inner_atlas, phi_AB, output_path + "/TC_mesh", False)
        tc_outer_patient = transform_mesh(tc_outer_atlas, phi_AB, output_path + "/TC_mesh", False)
        fc_inner, fc_mesh = mp.get_distance(fc_inner_patient, fc_outer_patient)
        tc_inner, tc_mesh = mp.get_distance(tc_inner_patient, tc_outer_patient)
        if keep_intermediate_outputs:
            mp.write_vtk_mesh(fc_inner, output_path + "/FC_inner.vtk")
            mp.write_vtk_mesh(tc_inner, output_path + "/TC_inner.vtk")
    else:
        print("Computing the thickness map via distance transformation from mask edges")
        fc_thickness_image, fc_distance, fc_mask = compute_thickness(FC_prob)
        tc_thickness_image, tc_distance, tc_mask = compute_thickness(TC_prob)
        fc_mesh = sample_distance_from_image(fc_thickness_image, fc_mesh)
        tc_mesh = sample_distance_from_image(tc_thickness_image, tc_mesh)
        if keep_intermediate_outputs:
            itk.imwrite(fc_thickness_image, os.path.join(output_path, "FC_thickness_image.nrrd"), compression=True)
            itk.imwrite(tc_thickness_image, os.path.join(output_path, "TC_thickness_image.nrrd"), compression=True)

    if keep_intermediate_outputs:
        mp.write_vtk_mesh(fc_mesh, output_path + "/FC_outer.vtk")
        mp.write_vtk_mesh(tc_mesh, output_path + "/TC_outer.vtk")

    print("Transforming meshes into atlas space")
    fc_mesh_atlas = transform_mesh(fc_mesh, phi_BA, output_path + "/FC_mesh", False)
    tc_mesh_atlas = transform_mesh(tc_mesh, phi_BA, output_path + "/TC_mesh", False)
    if keep_intermediate_outputs:
        mp.write_vtk_mesh(fc_mesh_atlas, output_path + "/FC_mesh_atlas.vtk")
        mp.write_vtk_mesh(tc_mesh_atlas, output_path + "/TC_mesh_atlas.vtk")

    print("Mapping thickness from patient meshes to the atlas")
    mapped_mesh_fc = mp.map_attributes(fc_mesh_atlas, inner_mesh_fc_atlas)
    mapped_mesh_tc = mp.map_attributes(tc_mesh_atlas, inner_mesh_tc_atlas)
    if keep_intermediate_outputs:
        mp.write_vtk_mesh(mapped_mesh_fc, output_path + "/FC_mapped_mesh.vtk")
        mp.write_vtk_mesh(mapped_mesh_tc, output_path + "/TC_mapped_mesh.vtk")

    print("Projecting thickness to 2D")
    thickness_3d_to_2d(mapped_mesh_fc, mesh_type='FC', output_filename=output_path + '/FC_thickness')
    thickness_3d_to_2d(mapped_mesh_tc, mesh_type='TC', output_filename=output_path + '/TC_thickness')


if __name__ == "__main__":
    if len(sys.argv) > 1:
        oai_dir = sys.argv[1]
        test_cases = {
            f"{oai_dir}/OAIBaselineImages/results/0.C.2/9016304/20040917/10242106": {
                "laterality": "left", "name": "9016304_0M"},
            f"{oai_dir}/OAIBaselineImages/results/0.C.2/9016304/20040917/10242112": {
                "laterality": "right", "name": "9016304_0M"},
            f"{oai_dir}/OAIBaselineImages/results/0.C.2/9021791/20040922/10250903": {
                "laterality": "left", "name": "9021791_0M"},
            f"{oai_dir}/OAIBaselineImages/results/0.C.2/9021791/20040922/10250909": {
                "laterality": "right", "name": "9021791_0M"},
            f"{oai_dir}/OAIBaselineImages/results/0.C.2/9094865/20050204/10165606": {
                "laterality": "left", "name": "9094865_0M"},
            f"{oai_dir}/OAIBaselineImages/results/0.C.2/9094865/20050204/10165612": {
                "laterality": "right", "name": "9094865_0M"},
            f"{oai_dir}/OAI96MonthImages/results/10.C.1/9016304/20120831/13486606": {
                "laterality": "left", "name": "9016304_96M"},
            f"{oai_dir}/OAI96MonthImages/results/10.C.1/9016304/20120831/13486612": {
                "laterality": "right", "name": "9016304_96M"},
            f"{oai_dir}/OAI96MonthImages/results/10.C.1/9021791/20121016/13466523": {
                "laterality": "left", "name": "9021791_96M"},
            f"{oai_dir}/OAI96MonthImages/results/10.C.1/9021791/20121016/13466530": {
                "laterality": "right", "name": "9021791_96M"},
            f"{oai_dir}/OAI96MonthImages/results/10.C.1/9040390/20121219/13439303/": {
                "laterality": "left", "name": "9040390_96M"},
            f"{oai_dir}/OAI96MonthImages/results/10.C.1/9040390/20121219/13439309/": {
                "laterality": "right", "name": "9040390_96M"},
        }
    else:  # only run on the test case bundled with the code
        test_cases = {
            f"{DATA_DIR}/test_data/colab_case/image_preprocessed.nii.gz": {
                "laterality": "left", "name": "test_case"},
        }

    for case, case_info in test_cases.items():
        print(f"\nProcessing {case}")
        output_path = f"./OAI_results_2/{case_info['name']}_{case_info['laterality']}"
        analysis_pipeline(case, output_path, case_info['laterality'], keep_intermediate_outputs=True)
