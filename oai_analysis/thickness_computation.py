from enum import Enum

import itk
import numpy as np


class DistanceMapMethod(str, Enum):
    signed_maurer = 'signed_maurer'
    parabolic_morphology = 'parabolic_morphology'


def masked_distance(mask, method=DistanceMapMethod.parabolic_morphology):
    """Distance map in physical distance units to the edge of the mask from inside the mask."""
    if method is DistanceMapMethod.signed_maurer:
        distance = itk.signed_maurer_distance_map_image_filter(mask.astype(np.uint8), inside_is_positive=True)
    elif method is DistanceMapMethod.parabolic_morphology:
        distance = itk.morphological_signed_distance_transform_image_filter(mask.astype(np.uint8),
                                                                            inside_is_positive=True)
    masked_distance = itk.mask_image_filter(distance, mask_image=mask.astype(itk.SS))
    return masked_distance


def masked_distance_pixels(mask, method=DistanceMapMethod.parabolic_morphology):
    """Distance map in pixel units to the edge of the mask from inside the mask."""
    if method is DistanceMapMethod.signed_maurer:
        distance_pixels = itk.signed_maurer_distance_map_image_filter(mask.astype(np.uint8), inside_is_positive=True,
                                                                      use_image_spacing=False)
    elif method is DistanceMapMethod.parabolic_morphology:
        distance_pixels = itk.morphological_signed_distance_transform_image_filter(mask.astype(np.uint8),
                                                                                   inside_is_positive=True,
                                                                                   use_image_spacing=False)
    masked_distance_pixels = itk.mask_image_filter(distance_pixels, mask_image=mask.astype(itk.SS))
    return masked_distance_pixels


def propagate_thickness(distance, distance_pixels, mask, max_distance=1000.0):
    """Propagate the thickness from a masked distance map to boundaries of the mask.

    max_distance (pixel units) must be larger than the largest thickness.
    """
    n_dilations = int(np.ceil(np.max(distance_pixels)))
    thickness = distance
    for iteration in reversed(range(1, n_dilations)):
        distance_pixels_mask = itk.binary_threshold_image_filter(distance_pixels,
                                                                 lower_threshold=float(iteration),
                                                                 inside_value=max_distance,
                                                                 outside_value=0.0)
        masked_thickness = itk.mask_image_filter(thickness,
                                                 mask_image=distance_pixels_mask.astype(itk.SS),
                                                 masking_value=0)
        dilated = itk.grayscale_geodesic_dilate_image_filter(marker_image=masked_thickness,
                                                             mask_image=distance_pixels_mask.astype(itk.F),
                                                             run_one_iteration=True,
                                                             fully_connected=True,
                                                             ttype=(type(distance), type(distance)))
        masked_thickness_comp = np.where(np.asarray(distance_pixels_mask) != max_distance, np.asarray(thickness), 0)
        masked_thickness_comp = itk.image_from_array(masked_thickness_comp)
        masked_thickness_comp.CopyInformation(thickness)

        thickness = itk.add_image_filter(dilated, masked_thickness_comp)
    return thickness


def compute_thickness(cartilage_probability, method = DistanceMapMethod.parabolic_morphology):
    mask_inside_value = 10.0
    # a value larger than the maximum possible thickness in pixel units
    max_distance = 100.0

    # Compute a binary mask from the cartilage segmentation probability
    mask = itk.binary_threshold_image_filter(cartilage_probability, lower_threshold=0.5, inside_value=mask_inside_value,
                                             outside_value=0.0)

    # Compute the distance map to the edge of the mask from inside the mask.
    distance = masked_distance(mask, method=method)

    # Compute the distance map in pixels
    distance_pixels = masked_distance_pixels(mask, method=method)

    # This is used to determine how many local dilations are required
    # to propagate the thickness to the upper and lower cartilage bounds.
    n_dilations = int(np.ceil(np.max(distance_pixels)))

    # Propagate the thickness from a masked distance map to boundaries of the mask
    thickness = propagate_thickness(distance, distance_pixels, mask, max_distance=max_distance)

    return thickness, distance