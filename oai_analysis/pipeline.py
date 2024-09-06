import os
import pathlib

import numpy as np
import itk
import matplotlib.pyplot as plt
import icon_registration.itk_wrapper as itk_wrapper
import icon_registration.pretrained_models as pretrained_models

from analysis_object import AnalysisObject
import mesh_processing as mp


def transform_mesh(mesh, transform):
    """
    Transform the input mesh using the provided transform.

    :param mesh: input mesh
    :param transform: The modelling transform to use (the inverse of image resampling transform)
    :return: transformed mesh
    """
    ttype = itk.TransformMeshFilter[itk.Mesh[itk.SS, 3], itk.Mesh[itk.SS, 3], itk.Transform[itk.D, 3, 3]]
    filter = ttype()
    filter.SetInput(mesh)
    filter.SetTransform(transform)
    filter.Update()
    transformed_mesh = filter.GetOutput()
    return transformed_mesh


def analysis_pipeline(input_path, output_path):
    """
    Computes cartilage thickness for femur and tibia from knee MRI.

    :param input_path: path to the input image file, or path to input directory containing DICOM image series.
    :param output_path: path to the desired directory for outputs.
    """
    in_image = itk.imread(input_path, pixel_type=itk.F)

    # segment the femoral and tibial cartilage
    obj = AnalysisObject()
    FC_prob, TC_prob = obj.segment(in_image)

    # Get the thickness map for the meshes
    distance_inner_FC, distance_outer_FC = mp.get_thickness_mesh(FC_prob, mesh_type='FC')
    distance_inner_TC, distance_outer_TC = mp.get_thickness_mesh(TC_prob, mesh_type='TC')

    # Register the input image to the atlas
    model = pretrained_models.OAI_knees_registration_model()
    DATA_DIR = pathlib.Path(__file__).parent / "data"
    atlas_filename = DATA_DIR / "atlases/atlas_60_LEFT_baseline_NMI/atlas.nii.gz"
    in_image_D = in_image.astype(itk.D)
    atlas_image = itk.imread(atlas_filename, itk.D)

    phi_AB, phi_BA = itk_wrapper.register_pair(model, in_image_D, atlas_image)
    interpolator = itk.LinearInterpolateImageFunction.New(in_image_D)

    # transform the thickness measurements to the atlas space
    # we use modelling transform, which is the inverse of the image resampling transform
    transformed_mesh_FC = transform_mesh(distance_inner_FC, transform=phi_BA)
    transformed_mesh_TC = transform_mesh(distance_inner_TC, transform=phi_BA)

    # Get inner and outer meshes for the TC and FC atlas meshes
    prob_fc_atlas = itk.imread(DATA_DIR / "atlases/atlas_60_LEFT_baseline_NMI/atlas_fc.nii.gz")
    mesh_fc_atlas = mp.get_mesh(prob_fc_atlas)
    inner_mesh_fc_atlas, outer_mesh_fc_atlas = mp.split_mesh(mesh_fc_atlas, mesh_type='FC')
    prob_tc_atlas = itk.imread(DATA_DIR / "atlases/atlas_60_LEFT_baseline_NMI/atlas_tc.nii.gz")
    mesh_tc_atlas = mp.get_mesh(prob_tc_atlas)
    inner_mesh_tc_atlas, outer_mesh_tc_atlas = mp.split_mesh(mesh_tc_atlas, mesh_type='TC')

    # For mapping the thickness to the atlas mesh
    mapped_mesh_fc = mp.map_attributes(transformed_mesh_FC, inner_mesh_fc_atlas)
    mapped_mesh_tc = mp.map_attributes(transformed_mesh_TC, inner_mesh_tc_atlas)

    os.makedirs(output_path, exist_ok=True)

    # Project thickness to 2D
    x, y, t = mp.project_thickness(mapped_mesh_tc, mesh_type='TC')
    s = plt.scatter(x, y, c=t, vmin=0, vmax=4)
    cb = plt.colorbar(s)
    cb.set_label('Thickness TC')
    plt.axis('off')
    plt.draw()
    plt.savefig(output_path + '/thickness_TC.png')
    plt.show()

    x, y, t = mp.project_thickness(mapped_mesh_fc, mesh_type='FC')
    plt.figure(figsize=(8, 6))
    s = plt.scatter(x, y, c=t)
    cb = plt.colorbar(s)
    cb.set_label('Thickness FC', size=15)
    plt.axis('off')
    plt.draw()
    plt.savefig(output_path + '/thickness_FC.png')
    plt.show()

    out_image_path = os.path.join(output_path, "in_image.nrrd")
    itk.imwrite(in_image, out_image_path)  # for debugging


if __name__ == "__main__":
    test_cases = [
        r"M:\Dev\Osteoarthritis\OAI_analysis_2\oai_analysis\data\test_data\colab_case\image_preprocessed.nii.gz",
        r"M:\Dev\Osteoarthritis\NAS\OAIBaselineImages\results\0.C.2\9000798\20040924\10249506",
        r"M:\Dev\Osteoarthritis\NAS\OAIBaselineImages\results\0.C.2\9000798\20040924\10249512",
        r"M:\Dev\Osteoarthritis\NAS\OAIBaselineImages\results\0.C.2\9000798\20040924\10249516",
        r"M:\Dev\Osteoarthritis\NAS\OAIBaselineImages\results\0.C.2\9000798\20040924\10249517",
    ]

    for i, case in enumerate(test_cases):
        print(f"Processing case {case}")
        analysis_pipeline(case, f"./OAI_results/case_{i:03d}")
