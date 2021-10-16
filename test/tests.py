import unittest
import subprocess
import oai_analysis_2.utils
import sys
import pathlib

import itk

import oai_analysis_2.analysis_object
import oai_analysis_2.registration

import numpy as np

TEST_DATA_DIR = pathlib.Path(__file__).parent / "test_files"

def download_test_data():
    subprocess.run(["girder-client",  "--api-url", "https://data.kitware.com/api/v1", "localsync", "6145e1332fa25629b9b1b2f7", TEST_DATA_DIR], stdout=sys.stdout)

class TestOAIAnalysis(unittest.TestCase):

    def setUp(self):
        download_test_data()
        self.analysis_object = oai_analysis_2.analysis_object.AnalysisObject()



    def testSegmentation(self):
        input_image = itk.imread(str(TEST_DATA_DIR / "colab_case/image_preprocessed.nii.gz"))

        correct_FC_segmentation = itk.imread(str(TEST_DATA_DIR / "colab_case/FC_probmap.nii.gz"))
        correct_TC_segmentation = itk.imread(str(TEST_DATA_DIR / "colab_case/TC_probmap.nii.gz"))
        
        FC, TC = self.analysis_object.segment(input_image)

        #print(np.sum(itk.ComparisonImageFilter(FC, correct_FC_segmentation)))
        #print(np.sum(itk.ComparisonImageFilter(TC, correct_TC_segmentation)))


        self.assertFalse(np.sum(itk.ComparisonImageFilter(FC, correct_FC_segmentation)) > 1)
        self.assertFalse(np.sum(itk.ComparisonImageFilter(TC, correct_TC_segmentation)) > 1)

    def testRegistration(self):
        input_image = itk.imread(str(TEST_DATA_DIR / "colab_case/image_preprocessed.nii.gz"))

        correct_registration = itk.imread(str(TEST_DATA_DIR / "colab_case/avsm/inv_transform_to_atlas.nii.gz"))

        registration = self.analysis_object.register(input_image)
        
        self.assertFalse(np.sum(itk.ComparisonImageFilter(registration, correct_registration)) > 1)

class TestImports(unittest.TestCase):

    def testImportsCPU(self):
        self.analysis_object = oai_analysis_2.analysis_object.AnalysisObject()

class TestICONRegistration(unittest.TestCase):
    def setUp(self):
        download_test_data()
    def testRegistration(self):
        ICON_obj = oai_analysis_2.registration.ICON_Registration()

        image_A = itk.imread(str(TEST_DATA_DIR / "colab_case/image_preprocessed.nii.gz"))
        image_B = itk.imread(oai_analysis_2.utils.get_data_dir() + "/atlas_60_LEFT_baseline_NMI/atlas_image.nii.gz")

        ICON_obj.register(image_A, image_B)

if __name__ == "__main__":
    unittest.main()


        

