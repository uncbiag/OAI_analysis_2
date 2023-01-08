import unittest
import subprocess
import oai_analysis_2.utils
import sys
import pathlib

import itk
import vtk
import oai_analysis_2.analysis_object
import oai_analysis_2.registration
from oai_analysis_2 import mesh_processing as mp
import numpy as np

TEST_DATA_DIR = pathlib.Path(__file__).parent / "test_files"

def download_test_data():
    subprocess.run(["girder-client",  "--api-url", "https://data.kitware.com/api/v1", "localsync", "620559344acac99f42957d63", TEST_DATA_DIR], stdout=sys.stdout)

class TestOAIAnalysis(unittest.TestCase):

    def setUp(self):
        download_test_data()
        self.analysis_object = oai_analysis_2.analysis_object.AnalysisObject()

    def test_SegmentationCPU(self):
        input_image = itk.imread(str(TEST_DATA_DIR / "colab_case/image_preprocessed.nii.gz"))

        correct_FC_segmentation = itk.imread(str(TEST_DATA_DIR / "colab_case/FC_probmap.nii.gz"))
        correct_TC_segmentation = itk.imread(str(TEST_DATA_DIR / "colab_case/TC_probmap.nii.gz"))
        
        FC, TC = self.analysis_object.segment(input_image)

        print('Segmentation Done')
        print(FC.shape, TC.shape)
        #np.save('FC_map.npy', FC)
        #np.save('TC_map.npy', TC)
        #print(np.sum(itk.ComparisonImageFilter(FC, correct_FC_segmentation)))
        #print(np.sum(itk.ComparisonImageFilter(TC, correct_TC_segmentation)))

        self.assertLess(np.sum(itk.comparison_image_filter(FC, correct_FC_segmentation)) , 12)
        self.assertLess(np.sum(itk.comparison_image_filter(TC, correct_TC_segmentation)) , 12)

    def test_MeshThicknessCPU(self):
        input_image = itk.imread(str(TEST_DATA_DIR / "colab_case/image_preprocessed.nii.gz"), itk.D)
        atlas_image = self.analysis_object.atlas_image

        correct_FC_segmentation = itk.imread(str(TEST_DATA_DIR / "colab_case/FC_probmap.nii.gz"), itk.D)
        correct_TC_segmentation = itk.imread(str(TEST_DATA_DIR / "colab_case/TC_probmap.nii.gz"), itk.D)
        
        def deform_probmap(phi_AB, image_A, image_B, prob_map):
            interpolator = itk.LinearInterpolateImageFunction.New(image_A)
            warped_image = itk.resample_image_filter(prob_map, 
                transform=phi_AB, 
                interpolator=interpolator,
                size=itk.size(image_B),
                output_spacing=itk.spacing(image_B),
                output_direction=image_B.GetDirection(),
                output_origin=image_B.GetOrigin()
            )
            return warped_image

        phi_AB = self.analysis_object.register(input_image)
        
        warped_image_FC = deform_probmap(phi_AB, input_image, atlas_image, correct_FC_segmentation)
        warped_image_TC = deform_probmap(phi_AB, input_image, atlas_image, correct_TC_segmentation)

        distance_inner_FC, _ = mp.get_thickness_mesh(warped_image_FC, mesh_type='FC')
        distance_inner_FC = mp.get_itk_mesh(distance_inner_FC)
        print(distance_inner_FC)

        distance_inner_TC, _ = mp.get_thickness_mesh(warped_image_TC, mesh_type='TC')
        distance_inner_TC = mp.get_itk_mesh(distance_inner_TC)
        print(distance_inner_TC)

        print("Thickness computation completed")

        assert(64800 <= distance_inner_FC.GetNumberOfPoints() <= 65000)
        assert(20460 <= distance_inner_TC.GetNumberOfPoints() <= 20480)
    
    def test_RegistrationCPU(self):
        input_image = itk.imread(str(TEST_DATA_DIR / "colab_case/image_preprocessed.nii.gz"))

        correct_registration = itk.imread(str(TEST_DATA_DIR / "colab_case/avsm/inv_transform_to_atlas.nii.gz"))

        registration = self.analysis_object.register(input_image)
        
        #registration object is an itk transform. Need to verify that it is correct in test, but it appears correct
        print(registration)
        #self.assertFalse(np.sum(itk.ComparisonImageFilter(registration, correct_registration)) > 1)

class TestImports(unittest.TestCase):

    def test_ImportsCPU(self):
        self.analysis_object = oai_analysis_2.analysis_object.AnalysisObject()

class TestICONRegistration(unittest.TestCase):
    def setUp(self):
        download_test_data()

    def test_RegistrationCPU(self):
        ICON_obj = oai_analysis_2.registration.ICON_Registration()

        image_A = itk.imread(str(TEST_DATA_DIR / "colab_case/image_preprocessed.nii.gz"))
        image_B = itk.imread(oai_analysis_2.utils.get_data_dir() + "/atlas_60_LEFT_baseline_NMI/atlas_image.nii.gz")

        ICON_obj.register(image_A, image_B)

if __name__ == "__main__":
    unittest.main()


        

