# All Imports

import numpy as np
import os
import dask
from dask import delayed, compute, visualize
from dask.distributed import Client, progress, LocalCluster


def image_normalize(image, window_min_perc, window_max_perc, output_min, output_max):
    import itk
    from itk import IntensityWindowingImageFilter as IntensityWindowingImageFilter

    window_rescale = IntensityWindowingImageFilter[type(image), type(image)].New()
    image_array = itk.GetArrayFromImage(image)

    window_min = np.percentile(image_array, window_min_perc)
    window_max = np.percentile(image_array, window_max_perc)

    window_rescale.SetInput(image)
    window_rescale.SetOutputMaximum(output_max)
    window_rescale.SetOutputMinimum(output_min)
    window_rescale.SetWindowMaximum(window_max)
    window_rescale.SetWindowMinimum(window_min)
    window_rescale.Update()
    return window_rescale.GetOutput()


def readimage(image_path):
    import itk
    from itk import CastImageFilter as CastImageFilter
    import xarray as xr

    xr_dataset = xr.open_zarr(image_path)
    xr_array = xr_dataset.get("image")
    itk_image = itk.image_from_xarray(xr_array)

    cast_filter_type = CastImageFilter[type(itk_image), itk.Image[itk.F, 3]].New()
    cast_filter_type.SetInPlace(False)
    cast_filter_type.SetInput(itk_image)
    cast_filter_type.Update()
    itk_image = cast_filter_type.GetOutput()
    return itk_image


@delayed(nout=3)
def register_images_delayed(image_A, image_B):
    import itk
    from itk import Image as itkImage
    from itk import CastImageFilter as CastImageFilter
    import icon_registration
    import icon_registration.itk_wrapper as itk_wrapper
    import icon_registration.pretrained_models as pretrained_models
    import torch
    import xarray as xr
    import time
    import numpy as np
    import gc

    image_A = readimage(image_A)
    image_B = readimage(image_B)

    cast_filter_type = CastImageFilter[type(image_A), itkImage[itk.D, 3]].New()
    cast_filter_type.SetInPlace(False)
    cast_filter_type.SetInput(image_A)
    cast_filter_type.Update()
    image_A = cast_filter_type.GetOutput()

    cast_filter_type = CastImageFilter[type(image_B), itkImage[itk.D, 3]].New()
    cast_filter_type.SetInPlace(False)
    cast_filter_type.SetInput(image_B)
    cast_filter_type.Update()
    image_B = cast_filter_type.GetOutput()

    image_A = image_normalize(image_A, 0.1, 99.9, 0, 1)

    model = pretrained_models.OAI_knees_gradICON_model()
    if torch.cuda.is_available():
        model.cuda()
        torch.cuda.empty_cache()
    else:
        model.to("cpu")

    # Register the images
    phi_AB, phi_BA = itk_wrapper.register_pair(model, image_A, image_B)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return phi_AB, image_A, image_B


@delayed
def deform_probmap_delayed(phi_AB, image_A, image_B, prob, image_type="FC"):
    import itk
    from itk import LinearInterpolateImageFunction

    interpolator = LinearInterpolateImageFunction.New(image_A)
    warped_image = itk.resample_image_filter(
        prob,
        transform=phi_AB,
        interpolator=interpolator,
        size=itk.size(image_B),
        output_spacing=itk.spacing(image_B),
        output_direction=image_B.GetDirection(),
        output_origin=image_B.GetOrigin(),
    )

    return warped_image


@delayed(nout=1)
def get_thickness(warped_image, mesh_type):
    import itk
    import numpy as np
    from oai_analysis_2 import mesh_processing as mp

    distance_inner, _ = mp.get_thickness_mesh(warped_image, mesh_type=mesh_type)
    distance_inner_itk = mp.get_itk_mesh(distance_inner)
    return distance_inner_itk


@delayed(nout=2)
def segment_method(image_A):
    import itk
    from itk import IntensityWindowingImageFilter as IntensityWindowingImageFilter
    import oai_analysis_2
    import torch
    import os
    from os.path import exists
    from oai_analysis_2 import utils
    from oai_analysis_2.segmentation import segmenter
    import numpy as np
    import urllib.request
    import gc

    if torch.cuda.is_available():
        device = "cuda"
        torch.cuda.empty_cache()
    else:
        print("WARNING: CUDA NOT AVAILABLE, FALLING BACK TO CPU")
        device = "cpu"

    if not exists("segmentation_model.pth.tar"):
        print("Downloading segmentation model")
        urllib.request.urlretrieve(
            "https://github.com/uncbiag/OAI_analysis_2/blob/master/data/segmentation_model.pth.tar?raw=true",
            "segmentation_model.pth.tar",
        )

    if not exists("segmentation_train_config.pth.tar"):
        urllib.request.urlretrieve(
            "https://github.com/uncbiag/OAI_analysis_2/blob/master/data/segmentation_train_config.pth.tar?raw=true",
            "segmentation_train_config.pth.tar",
        )

    # Initialize segmenter
    segmenter_config = dict(
        ckpoint_path="segmentation_model.pth.tar",
        training_config_file="segmentation_train_config.pth.tar",
        device=device,
        batch_size=2,
        overlap_size=(16, 16, 8),
        output_prob=True,
        output_itk=True,
    )

    segmenter = oai_analysis_2.segmentation.segmenter.Segmenter3DInPatchClassWise(
        mode="pred", config=segmenter_config
    )

    # Segment downloaded image
    test_volume = readimage(image_A)

    test_volume = image_normalize(test_volume, 0.1, 99.9, 0, 1)

    FC_probmap, TC_probmap = segmenter.segment(
        test_volume, if_output_prob_map=True, if_output_itk=True
    )

    del segmenter
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return FC_probmap, TC_probmap
