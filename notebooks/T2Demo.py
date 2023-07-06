# image_A: fixed image, image_B: moving image
import os
import itk
import torch
import numpy as np
import matplotlib.pyplot as plt
import icon_registration.itk_wrapper as itk_wrapper
import icon_registration.pretrained_models as pretrained_models

import oai_analysis_2.mesh_processing as mp
from oai_analysis_2.analysis_object import AnalysisObject

from oai_analysis_2.T2_process import read_t2, calculate_t2, get_t2_mesh, np2itk, project_statistics, register_pair_with_mask
from oai_analysis_2.affine_models import OAI_affine_model

import warnings
warnings.filterwarnings("ignore")
torch.manual_seed(123)
np.random.seed(123)


def plot_image(images, image_names, save_name, cmap=None):
    assert len(images) == len(image_names)
    height = int(np.ceil(len(images) / 2))
    width = 2 if len(images) > 1 else 1
    plt.clf()
    for i in range(len(images)):
        plt.subplot(height, width, i + 1)
        ax = plt.gca()
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
        plt.title(image_names[i])
        plt.imshow(images[i], cmap=cmap)
    plt.savefig(save_name)

volumes = 7
finetune_steps = 15
measures = ['mean', 'median']

path_DESS = "./OAIData/9298954_RIGHT_00m/image_preprocessed.nii.gz"
path_T2 = "./OAIData/9298954_RIGHT_00m/SAG_T2_MAP"

path_atlas = "./OAIData/atlas_image.nii.gz"

result_folder = os.path.join('./result', path_DESS.split('/')[2])
os.makedirs(result_folder, exist_ok=True)

### step1: DESS segmentation ###
image_DESS = itk.imread(path_DESS)
obj = AnalysisObject()
FC_prob, TC_prob = obj.segment(image_DESS)
# Visualize
images = [FC_prob[100], TC_prob[100]]
image_names = ["FC probmap", "TC probmap"]
plot_image(images, image_names, os.path.join(result_folder, 'segmentation.png'), cmap='gray')
###

### step2: Register smaps to DESS ###
image_DESS = itk.imread(path_DESS, itk.D)
image_T2, image_mask = read_t2(image_path=path_T2, ref_path=path_DESS, side="RIGHT", smap_folder="./smap")
model = OAI_affine_model()
_, phi_BA = register_pair_with_mask(model, image_A=image_DESS, image_B=image_T2[0], image_B_mask=image_mask, finetune_steps=finetune_steps)
interpolator = itk.NearestNeighborInterpolateImageFunction.New(image_T2[0])
warped_image_0 = itk.resample_image_filter(image_T2[0], transform=phi_BA, interpolator=interpolator,
                                         size=itk.size(image_DESS), output_spacing=itk.spacing(image_DESS),
                                         output_direction=image_DESS.GetDirection(), output_origin=image_DESS.GetOrigin())
# Visualize
checker_board = itk.checker_board_image_filter(warped_image_0, image_DESS)
images = [image_T2[0][80], image_DESS[80], warped_image_0[80], checker_board[80]]
image_names = ["Moving Image", "Fixed Image", "Warped Image", "Checker Board"]
plot_image(images, image_names, os.path.join(result_folder, 'DESS-T2 registration.png'))

warped_image = []
for i in range(1, volumes):
    warped_image_Ai = itk.resample_image_filter(image_T2[i], transform=phi_BA, interpolator=interpolator,
                                                size=itk.size(image_DESS), output_spacing=itk.spacing(image_DESS),
                                                output_direction=image_DESS.GetDirection(), output_origin=image_DESS.GetOrigin())
    warped_image_Ai = itk.GetArrayViewFromImage(warped_image_Ai)
    warped_image.append(warped_image_Ai)

### step3: Calculate T2 values ###
np_t2_FC, np_s0_FC = calculate_t2(warped_image, itk.GetArrayViewFromImage(FC_prob))
np_t2_TC, np_s0_TC = calculate_t2(warped_image, itk.GetArrayViewFromImage(TC_prob))
t2_FC = np2itk(np_t2_FC, image_DESS)
t2_TC = np2itk(np_t2_TC, image_DESS)

### step4: Register DESS to Atlas ###
image_atlas = itk.imread(path_atlas,  itk.D)
model = pretrained_models.OAI_knees_registration_model()
phi_AB, phi_BA = itk_wrapper.register_pair(model, image_DESS, image_atlas)
interpolator = itk.LinearInterpolateImageFunction.New(image_DESS)
# Visualize
warped_image_DESS = itk.resample_image_filter(image_DESS, transform=phi_AB, interpolator=interpolator,
                                              size=itk.size(image_atlas), output_spacing=itk.spacing(image_atlas),
                                              output_direction=image_atlas.GetDirection(), output_origin=image_atlas.GetOrigin())
checker_board = itk.checker_board_image_filter(warped_image_DESS, image_atlas)
images = [image_DESS[80], image_atlas[80], warped_image_DESS[80], checker_board[80]]
image_names = ["Moving Image", "Fixed Image", "Warped Image", "Checker Board"]
plot_image(images, image_names, os.path.join(result_folder, 'altas-DESS registration.png'))

### step4: Deform T2 using the transform obtained from step4 ###
warped_image_FC_T2 = itk.resample_image_filter(t2_FC, transform=phi_AB, interpolator=interpolator,
                                               size=itk.size(image_atlas), output_spacing=itk.spacing(image_atlas),
                                               output_direction=image_atlas.GetDirection(), output_origin=image_atlas.GetOrigin())
warped_image_TC_T2 = itk.resample_image_filter(t2_TC, transform=phi_AB, interpolator=interpolator,
                                               size=itk.size(image_atlas), output_spacing=itk.spacing(image_atlas),
                                               output_direction=image_atlas.GetDirection(), output_origin=image_atlas.GetOrigin())
warped_image_FC_prob = itk.resample_image_filter(FC_prob, transform=phi_AB, interpolator=interpolator,
                                                 size=itk.size(image_atlas), output_spacing=itk.spacing(image_atlas),
                                                 output_direction=image_atlas.GetDirection(), output_origin=image_atlas.GetOrigin())
warped_image_TC_prob = itk.resample_image_filter(TC_prob, transform=phi_AB, interpolator=interpolator,
                                                 size=itk.size(image_atlas), output_spacing=itk.spacing(image_atlas),
                                                 output_direction=image_atlas.GetDirection(), output_origin=image_atlas.GetOrigin())

### step5: Map T2 values on meshes ###
t2_inner_FC = get_t2_mesh(warped_image_FC_T2, warped_image_FC_prob, mesh_type='FC', measures=measures)
t2_inner_TC = get_t2_mesh(warped_image_TC_T2, warped_image_TC_prob, mesh_type='TC', measures=measures)

### step6: Get inner and outer meshes for TC & FC atlas ###
prob_fc_atlas = itk.imread('./OAIData/atlas_fc.nii.gz')
mesh_fc_atlas = mp.get_mesh(prob_fc_atlas)
inner_mesh_fc_atlas, outer_mesh_fc_atlas = mp.split_mesh(mesh_fc_atlas, mesh_type='FC')

prob_tc_atlas = itk.imread('./OAIData/atlas_tc.nii.gz')
mesh_tc_atlas = mp.get_mesh(prob_tc_atlas)
inner_mesh_tc_atlas, outer_mesh_tc_atlas = mp.split_mesh(mesh_tc_atlas, mesh_type='TC')

### step7: Map T2 values to the atlas mesh ###
mapped_mesh_fc = mp.map_attributes(t2_inner_FC, inner_mesh_fc_atlas)
mapped_mesh_tc = mp.map_attributes(t2_inner_TC, inner_mesh_tc_atlas)
# Visualize
meshes = {'FC': mapped_mesh_fc, 'TC': mapped_mesh_tc}
for i in range(len(measures)):
    # Plot the 2D Thickness projection for FC & TC
    for mesh_type in list(meshes.keys()):
        plt.clf()
        x, y, t = project_statistics(meshes[mesh_type], mesh_type=mesh_type, measure_num=i)
        s = plt.scatter(x, y, c=t, vmin=0, vmax=150)
        cb = plt.colorbar(s)
        cb.set_label('T2 ' + measures[i] + ' ' +mesh_type)
        plt.axis('off')
        plt.draw()
        plt.savefig(os.path.join(result_folder, 'T2_' + measures[i] + '_' + mesh_type + '.png'))
