from enum import Enum

import itk
import numpy as np


class DistanceMapMethod(str, Enum):
    signed_maurer = 'signed_maurer'
    parabolic_morphology = 'parabolic_morphology'


def distance_mm(mask, method=DistanceMapMethod.parabolic_morphology):
    """Distance map in physical distance units to the edge of the mask from inside the mask."""
    if method is DistanceMapMethod.signed_maurer:
        distance = itk.signed_maurer_distance_map_image_filter(mask.astype(np.uint8), inside_is_positive=True)
    elif method is DistanceMapMethod.parabolic_morphology:
        distance = itk.morphological_signed_distance_transform_image_filter(mask.astype(np.uint8),
                                                                            inside_is_positive=True)
    return distance


def distance_pixels(mask, method=DistanceMapMethod.parabolic_morphology):
    """Distance map in pixel units to the edge of the mask from inside the mask."""
    if method is DistanceMapMethod.signed_maurer:
        distance = itk.signed_maurer_distance_map_image_filter(mask.astype(np.uint8), inside_is_positive=True,
                                                               use_image_spacing=False)
    elif method is DistanceMapMethod.parabolic_morphology:
        distance = itk.morphological_signed_distance_transform_image_filter(mask.astype(np.uint8),
                                                                            inside_is_positive=True,
                                                                            use_image_spacing=False)
    return distance


def propagate_thickness(distance, distance_pixels, n_dilations, max_distance=250.0):
    """Propagate the thickness from a masked distance map to boundaries of the mask.

    max_distance (pixel units) must be larger than the largest thickness.
    """
    thickness = distance
    for iteration in reversed(range(1, n_dilations)):
        distance_pixels_mask = itk.binary_threshold_image_filter(distance_pixels,
                                                                 lower_threshold=float(iteration),
                                                                 inside_value=max_distance,
                                                                 outside_value=0.0)
        # itk.imwrite(distance_pixels_mask, f"distance_pixels_mask_{iteration}-label.nrrd", compression=True)

        masked_thickness = itk.mask_image_filter(thickness,
                                                 mask_image=distance_pixels_mask.astype(itk.SS),
                                                 masking_value=0)
        # itk.imwrite(masked_thickness, f"masked_thickness_{iteration}.nrrd", compression=True)

        dilated = itk.grayscale_geodesic_dilate_image_filter(marker_image=masked_thickness,
                                                             mask_image=distance_pixels_mask.astype(itk.F),
                                                             run_one_iteration=True,
                                                             fully_connected=True,
                                                             ttype=(type(distance), type(distance)))
        # itk.imwrite(dilated, f"dilated_{iteration}.nrrd", compression=True)

        masked_thickness_comp = np.where(np.asarray(distance_pixels_mask) != max_distance, np.asarray(thickness), 0)
        masked_thickness_comp = itk.image_from_array(masked_thickness_comp)
        masked_thickness_comp.CopyInformation(thickness)
        # itk.imwrite(masked_thickness_comp, f"masked_thickness_comp_{iteration}.nrrd", compression=True)

        thickness = itk.add_image_filter(dilated, masked_thickness_comp)
        # itk.imwrite(thickness, f"thickness_{iteration}.nrrd", compression=True)
    return thickness


def compute_thickness(cartilage_probability, method=DistanceMapMethod.parabolic_morphology):
    mask_inside_value = 10.0
    # a value larger than the maximum possible thickness in pixel units
    max_distance = 100.0
    expansion_padding = 5.0  # how many millimeters of extra propagation to use

    # Compute a binary mask from the cartilage segmentation probability
    mask = itk.binary_threshold_image_filter(
        cartilage_probability, lower_threshold=0.5, inside_value=mask_inside_value, outside_value=0.0)

    # Compute the distance map to the edge of the mask from inside the mask.
    distance = distance_mm(mask, method=method)
    enlarged_mask = itk.binary_threshold_image_filter(
        distance, lower_threshold=-expansion_padding, inside_value=mask_inside_value)
    distance = itk.multiply_image_filter(distance, 2.0)  # convert to thickness
    distance_padded = itk.add_image_filter(distance, expansion_padding * 2)  # make it strictly positive
    masked_distance = itk.mask_image_filter(distance_padded, mask_image=enlarged_mask.astype(itk.SS))
    # itk.imwrite(masked_distance, "masked_distance.nrrd", compression=True)

    # Compute the distance map in pixels
    distance_px = distance_pixels(mask, method=method)
    masked_distance_px = itk.mask_image_filter(distance_px, mask_image=enlarged_mask.astype(itk.SS))
    inside_dist= int(np.ceil(np.max(distance_px)))
    outside_dist = int(-np.floor(np.min(masked_distance_px)))
    n_dilations = inside_dist + outside_dist
    distance_px_padded = itk.add_image_filter(distance_px, outside_dist)  # make it strictly positive
    masked_distance_px = itk.mask_image_filter(distance_px_padded, mask_image=enlarged_mask.astype(itk.SS))
    # itk.imwrite(masked_distance_px, "masked_distance_px.nrrd", compression=True)

    # Propagate the thickness from a masked distance map to boundaries of the mask
    thickness = propagate_thickness(masked_distance, masked_distance_px, n_dilations, max_distance=max_distance)
    # now remove the padding
    thickness = itk.subtract_image_filter(thickness, expansion_padding * 2)
    thickness = itk.mask_image_filter(thickness, mask_image=enlarged_mask.astype(itk.SS))

    return thickness, distance
