import itk
import torch
import oai_analysis.segmentation.segmenter
import oai_analysis.registration
from oai_analysis.data import atlases_dir, models_dir

import os

class AnalysisObject:
    def __init__(self):
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            print("WARNING: CUDA NOT AVAILABLE, FALLING BACK TO CPU")
            self.device = "cpu"

        ## Initialize segmenter
        segmenter_config = dict(
            ckpoint_path=str(models_dir() / "segmentation_model.pth.tar"),
            training_config_file=str(models_dir() / "segmentation_train_config.pth.tar"),
            device=self.device,
            batch_size=4,
            overlap_size=(16, 16, 8),
            output_prob=True,
            output_itk=True,
        )
        self.segmenter = oai_analysis.segmentation.segmenter.Segmenter3DInPatchClassWise(
            mode="pred", config=segmenter_config
        )

        ## Initialize registerer
        #self.registerer = oai_analysis.registration.AVSM_Registration(
        #    from oai_analysis.test_data import data_dir
        #    ckpoint_path = data_dir() / "pre_trained_registration_model"
        #    config_path = data_dir() / "avsm_settings"
        #

        self.registerer = oai_analysis.registration.ICON_Registration()

        ## Load Atlas
        self.atlas_image = itk.imread(atlases_dir() / "atlas_60_LEFT_baseline_NMI" / "atlas_image.nii.gz")

    def segment(self, preprocessed_image):
        FC_probmap, TC_probmap = self.segmenter.segment(preprocessed_image, if_output_prob_map=True, if_output_itk=True)
        return (FC_probmap, TC_probmap)

    def register(self, preprocessed_image):
        registration = self.registerer.register(preprocessed_image, self.atlas_image)
        return registration
