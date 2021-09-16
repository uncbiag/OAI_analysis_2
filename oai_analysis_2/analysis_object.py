import itk
import torch
from oai_analysis_2 import utils
import oai_analysis_2.segmentation.segmenter

import os

class AnalysisObject:
    def __init__(self):
        if "cuda" in torch.testing.get_all_device_types():
            self.device = "cuda"
        else:
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

    def segment(self, preprocessed_image):
        FC_probmap, TC_probmap = self.segmenter.segment(preprocessed_image, if_output_prob_map=True, if_output_itk=True)
        return (FC_probmap, TC_probmap)
