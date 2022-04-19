import itk
import torch
from oai_analysis_2 import utils
import oai_analysis_2.segmentation.segmenter
import oai_analysis_2.registration

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
            ckpoint_path=os.path.join(utils.get_data_dir(), "segmentation_model.pth.tar"),
            training_config_file=os.path.join(utils.get_data_dir()
            , "segmentation_train_config.pth.tar"),
            device=self.device,
            batch_size=4,
            overlap_size=(16, 16, 8),
            output_prob=True,
            output_itk=True,
        )
        self.segmenter = oai_analysis_2.segmentation.segmenter.Segmenter3DInPatchClassWise(
            mode="pred", config=segmenter_config
        )

        ## Initialize registerer
        #self.registerer = oai_analysis_2.registration.AVSM_Registration(
        #    ckpoint_path=os.path.join(utils.get_data_dir(), "pre_trained_registration_model"),
        #    config_path =os.path.join(utils.get_data_dir(), "avsm_settings")
        #

        self.registerer = oai_analysis_2.registration.ICON_Registration()

        ## Load Atlas
        self.atlas_image = itk.imread(os.path.join(utils.get_data_dir(), "atlas_60_LEFT_baseline_NMI/atlas_image.nii.gz"))

    def segment(self, preprocessed_image):
        FC_probmap, TC_probmap = self.segmenter.segment(preprocessed_image, if_output_prob_map=True, if_output_itk=True)
        return (FC_probmap, TC_probmap)

    def register(self, preprocessed_image):
        registration = self.registerer.register(preprocessed_image, self.atlas_image)
        return registration
