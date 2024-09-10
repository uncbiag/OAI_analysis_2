import os
import pathlib

import icon_registration.itk_wrapper as itk_wrapper
import itk
import matplotlib.pyplot as plt
import vtk
from unigradicon import preprocess, get_unigradicon

import mesh_processing as mp
from analysis_object import AnalysisObject


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

    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(filename_prefix + "_transformed.vtk")
    reader.Update()
    transformed_mesh = reader.GetOutput()

    if keep_intermediate_outputs:
        write_vtk_mesh(mesh, filename_prefix + "_original.vtk")
    else:
        os.remove(filename_prefix + "_transformed.vtk")

    return transformed_mesh


def analysis_pipeline(input_path, output_path, keep_intermediate_outputs):
    """
    Computes cartilage thickness for femur and tibia from knee MRI.

    :param input_path: path to the input image file, or path to input directory containing DICOM image series.
    :param output_path: path to the desired directory for outputs.
    """
    in_image = itk.imread(input_path, pixel_type=itk.F)
    in_image = preprocess(in_image, modality="mri")
    os.makedirs(output_path, exist_ok=True)  # also holds intermediate results
    if keep_intermediate_outputs:
        itk.imwrite(in_image, os.path.join(output_path, "in_image.nrrd"))

    print("Segmenting the femoral and tibial cartilage")
    obj = AnalysisObject()
    FC_prob, TC_prob = obj.segment(in_image)
    if keep_intermediate_outputs:
        print("Saving segmentation results")
        itk.imwrite(FC_prob.astype(itk.F), os.path.join(output_path, "FC_prob.nrrd"), compression=True)
        itk.imwrite(TC_prob.astype(itk.F), os.path.join(output_path, "TC_prob.nrrd"), compression=True)

    print("Computing the thickness map for the meshes")
    distance_inner_FC, distance_outer_FC = mp.get_thickness_mesh(FC_prob, mesh_type='FC')
    distance_inner_TC, distance_outer_TC = mp.get_thickness_mesh(TC_prob, mesh_type='TC')

    print("Registering the input image to the atlas")
    model = get_unigradicon()
    DATA_DIR = pathlib.Path(__file__).parent / "data"
    atlas_filename = DATA_DIR / "atlases/atlas_60_LEFT_baseline_NMI/atlas.nii.gz"
    in_image_D = in_image.astype(itk.D)
    atlas_image = itk.imread(atlas_filename, itk.D)

    phi_AB, phi_BA = itk_wrapper.register_pair(model, in_image_D, atlas_image, finetune_steps=None)
    if keep_intermediate_outputs:
        print("Saving registration results")
        itk.transformwrite(phi_AB, os.path.join(output_path, "resampling.tfm"))
        itk.transformwrite(phi_BA, os.path.join(output_path, "modelling.tfm"))

    print("Transforming the thickness measurements to the atlas space")
    # we use modelling transform, which is the inverse of the image resampling transform
    transformed_mesh_FC = transform_mesh(distance_inner_FC, transform=phi_BA, filename_prefix=output_path + "/FC",
                                         keep_intermediate_outputs=keep_intermediate_outputs)
    transformed_mesh_TC = transform_mesh(distance_inner_TC, transform=phi_BA, filename_prefix=output_path + "/TC",
                                         keep_intermediate_outputs=keep_intermediate_outputs)

    print("Getting inner and outer meshes for the TC and FC atlas meshes")
    prob_fc_atlas = itk.imread(DATA_DIR / "atlases/atlas_60_LEFT_baseline_NMI/atlas_fc.nii.gz")
    mesh_fc_atlas = mp.get_mesh(prob_fc_atlas)
    inner_mesh_fc_atlas, outer_mesh_fc_atlas = mp.split_mesh(mesh_fc_atlas, mesh_type='FC')
    write_vtk_mesh(inner_mesh_fc_atlas, output_path + "/inner_mesh_fc_atlas.vtk")
    prob_tc_atlas = itk.imread(DATA_DIR / "atlases/atlas_60_LEFT_baseline_NMI/atlas_tc.nii.gz")
    mesh_tc_atlas = mp.get_mesh(prob_tc_atlas)
    inner_mesh_tc_atlas, outer_mesh_tc_atlas = mp.split_mesh(mesh_tc_atlas, mesh_type='TC')
    write_vtk_mesh(inner_mesh_tc_atlas, output_path + "/inner_mesh_tc_atlas.vtk")

    print("Mapping the thickness to the atlas mesh")
    mapped_mesh_fc = mp.map_attributes(transformed_mesh_FC, inner_mesh_fc_atlas)
    mapped_mesh_tc = mp.map_attributes(transformed_mesh_TC, inner_mesh_tc_atlas)

    print("Projecting thickness to 2D")
    x, y, t = mp.project_thickness(mapped_mesh_tc, mesh_type='TC')
    s = plt.scatter(x, y, c=t, vmin=0, vmax=4)
    cb = plt.colorbar(s)
    cb.set_label('Thickness TC')
    plt.axis('off')
    plt.draw()
    plt.savefig(output_path + '/thickness_TC.png')

    x, y, t = mp.project_thickness(mapped_mesh_fc, mesh_type='FC')
    plt.figure(figsize=(8, 6))
    s = plt.scatter(x, y, c=t)
    cb = plt.colorbar(s)
    cb.set_label('Thickness FC', size=15)
    plt.axis('off')
    plt.draw()
    plt.savefig(output_path + '/thickness_FC.png')


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
        analysis_pipeline(case, output_path=f"./OAI_results/case_{i:03d}", keep_intermediate_outputs=True)
