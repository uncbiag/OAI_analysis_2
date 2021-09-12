import unittest

import pathlib

import itk

import oai_analysis_2.analysis_object

TEST_DATA_DIR = pathlib.Path(__file__).parent / "test_files"

class TestOAIAnalysis(unittest.TestCase):

    def setUp(self):
        self.analysis_object = oai_analysis_2.analysis_object.AnalysisObject()

    def testSegmentation(self):
        input_image = itk.imread(str(TEST_DATA_DIR / "colab_case/image_preprocessed.nii.gz"))

        correct_FC_segmentation = itk.imread(str(TEST_DATA_DIR / "colab_case/FC_probmap.nii.gz"))
        correct_TC_segmentation = itk.imread(str(TEST_DATA_DIR / "colab_case/TC_probmap.nii.gz"))
        
        FC, TC = self.analysis_object.segment(input_image)


if __name__ == "__main__":
    unittest.main()


        

