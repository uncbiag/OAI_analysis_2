import os
import torch
import numpy as np
import urllib.request
from torch import nn
from collections import namedtuple
from icon_registration import config
import icon_registration.networks as networks
import icon_registration.network_wrappers as network_wrappers

from icon_registration.mermaidlite import compute_warped_image_multiNC
from icon_registration.losses import flips


def compute_marginal_entropy(values, bins, sigma, normalizer_1d):

    p = torch.exp(-((values - bins).pow(2).div(sigma))).div(normalizer_1d)
    p_n = p.mean(dim=1)
    p_n = p_n/(torch.sum(p_n) + 1e-10)

    return -(p_n * torch.log2(p_n + 1e-10)).sum(), p


def NMI_loss(warped_image, fixed_image, image_mask=None, bins=16, sigma=0.1, spatial_samples=0.1):
    new_sigma = 2 * sigma ** 2
    normalizer_1d = np.sqrt(2.0 * np.pi) * sigma
    normalizer_2d = 2.0 * np.pi * sigma ** 2

    background_fixed = torch.mean(fixed_image)
    background_warped = torch.mean(warped_image)
    max_fixed = torch.max(fixed_image)
    max_warped = torch.max(warped_image)

    bins_fixed_image = torch.linspace(background_fixed.item(), max_fixed.item(), bins,
                                            device=fixed_image.device, dtype=fixed_image.dtype).unsqueeze(1)

    max_m_f = torch.max(max_fixed, max_warped)
    bins_warped_image = torch.linspace(background_warped.item(), max_m_f.item(), bins,
                                          device=fixed_image.device, dtype=fixed_image.dtype).unsqueeze(1)

    if image_mask is not None:
        moving_image_valid = torch.where((image_mask < 0.5), torch.tensor(0, dtype=torch.float).to(warped_image.device), warped_image)
        fixed_image_valid = torch.where((image_mask < 0.5), torch.tensor(0, dtype=torch.float).to(warped_image.device), fixed_image)
    else:
        moving_image_valid = warped_image
        fixed_image_valid = fixed_image

    mask = (fixed_image_valid > background_fixed) & (moving_image_valid > background_warped)

    fixed_image = torch.masked_select(fixed_image, mask)
    warped_image = torch.masked_select(warped_image, mask)
    number_of_pixel = warped_image.shape[0]
    sample = torch.zeros(number_of_pixel, device=fixed_image.device,
                      dtype=fixed_image.dtype).uniform_() < spatial_samples

    # compute marginal entropy fixed image
    fixed_image = torch.masked_select(fixed_image.view(-1), sample)
    ent_fixed_image, p_f = compute_marginal_entropy(fixed_image, bins_fixed_image, new_sigma, normalizer_1d)

    # compute marginal entropy warped image
    warped_image = torch.masked_select(warped_image.view(-1), sample)
    ent_warped_image, p_m = compute_marginal_entropy(warped_image, bins_warped_image, new_sigma, normalizer_1d)

    # compute joint entropy
    p_joint = torch.mm(p_f, p_m.transpose(0, 1)).div(normalizer_2d)
    p_joint = p_joint / (torch.sum(p_joint) + 1e-10)

    ent_joint = -(p_joint * torch.log2(p_joint + 1e-10)).sum()

    # return -(ent_fixed_image + ent_warped_image - ent_joint)  # MI
    return -(ent_fixed_image + ent_warped_image)/ent_joint  # NMI


def OAI_affine_model(pretrained=True):
    # The definition of our affine registration network
    net = InverseConsistentNetWithMask(
        network_wrappers.FunctionFromMatrix(networks.ConvolutionalMatrixNet(dimension=3)),
        NMI_loss,
        2000
    )
    BATCH_SIZE = 4
    SCALE = 2 # 1 IS QUARTER RES, 2 IS HALF RES, 4 IS FULL RES
    input_shape = [BATCH_SIZE, 1, 40 * SCALE, 96 * SCALE, 96 * SCALE]

    # network_wrappers.assignIdentityMap(net, input_shape)
    net.assign_identity_map(input_shape)
    net.to(config.device)
    if pretrained:
        weights_location = "./network_weights/model_best.pth.tar"
        if not os.path.exists(weights_location):
            print("weight missing")
            return net
        trained_weights = torch.load(weights_location, map_location=torch.device("cpu"))
        net.load_state_dict(trained_weights['model'])
    return net


ICONLoss = namedtuple(
    "ICONLoss",
    "all_loss inverse_consistency_loss similarity_loss transform_magnitude flips",
)

class InverseConsistentNetWithMask(network_wrappers.RegistrationModule):
    def __init__(self, network, similarity, lmbda):

        super().__init__()

        self.regis_net = network
        self.lmbda = lmbda
        self.similarity = similarity

    def forward(self, image_A, image_B, image_A_mask=None, image_B_mask=None):

        assert self.identity_map.shape[2:] == image_A.shape[2:]
        assert self.identity_map.shape[2:] == image_B.shape[2:]

        # Tag used elsewhere for optimization.
        # Must be set at beginning of forward b/c not preserved by .cuda() etc
        self.identity_map.isIdentity = True

        self.phi_AB = self.regis_net(image_A, image_B)
        self.phi_BA = self.regis_net(image_B, image_A)

        self.phi_AB_vectorfield = self.phi_AB(self.identity_map)
        self.phi_BA_vectorfield = self.phi_BA(self.identity_map)

        # tag images during warping so that the similarity measure
        # can use information about whether a sample is interpolated
        # or extrapolated

        inbounds_tag = torch.zeros(tuple(image_A.shape), device=image_A.device)
        if len(self.input_shape) - 2 == 3:
            inbounds_tag[:, :, 1:-1, 1:-1, 1:-1] = 1.0
        elif len(self.input_shape) - 2 == 2:
            inbounds_tag[:, :, 1:-1, 1:-1] = 1.0
        else:
            inbounds_tag[:, :, 1:-1] = 1.0

        self.warped_image_A = compute_warped_image_multiNC(
            torch.cat([image_A, inbounds_tag], axis=1),
            self.phi_AB_vectorfield,
            self.spacing,
            1,
        )
        self.warped_image_B = compute_warped_image_multiNC(
            torch.cat([image_B, inbounds_tag], axis=1),
            self.phi_BA_vectorfield,
            self.spacing,
            1,
        )
        self.warped_image_A_mask = compute_warped_image_multiNC(
            torch.cat([image_A_mask, inbounds_tag], axis=1),
            self.phi_AB_vectorfield,
            self.spacing,
            1,
        ) if image_A_mask is not None else None
        self.warped_image_B_mask = compute_warped_image_multiNC(
            torch.cat([image_B_mask, inbounds_tag], axis=1),
            self.phi_BA_vectorfield,
            self.spacing,
            1,
        ) if image_B_mask is not None else None

        return (self.similarity(self.warped_image_B, image_A, self.warped_image_B_mask, image_A_mask), )
