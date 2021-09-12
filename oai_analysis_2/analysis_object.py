import itk

from oai_analysis_2 import utils
import oai_analysis_2.segmentation.segmenter

import os

class AnalysisObject:
    def __init__(self):
        ## Initialize segmenter

        segmenter_config = dict(
            ckpoint_path=os.path.join(utils.get_data_dir(), "segmentation_model.pth.tar"),
            training_config_file=os.path.join(utils.get_data_dir()
            , "segmentation_train_config.pth.tar"),
            device="cuda",
            batch_size=4,
            overlap_size=(16, 16, 8),
            output_prob=True,
            output_itk=True,
        )
        print(segmenter_config)
        self.segmenter = oai_analysis_2.segmentation.segmenter.Segmenter3DInPatchClassWise(
            mode="pred", config=segmenter_config
        )

    def segment(self, preprocessed_image):
        FC_probmap, TC_probmap = self.segmenter.segment(preprocessed_image, if_output_prob_map=True, if_output_itk=True)
        return (FC_probmap, TC_probmap)
