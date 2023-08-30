#!/usr/bin/env python
"""
Created by zhenlinx on 1/19/19
"""
from abc import ABC, abstractmethod
import torch
import numpy as np

from .image_transforms import Partition
from .utils import initialize_model
from .module_parameters import ParameterDict
from .networks import get_network

def load_json_to_dict(json_file):
    para = ParameterDict()
    para.load_JSON(json_file)
    return para.ext

class Segmenter(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        self.model = None
        self.config = None

    @abstractmethod
    def train(self, *args, **kwargs):
        pass

    @abstractmethod
    def test(self, *args, **kwargs):
        pass

    @abstractmethod
    def segment(self, *args, **kwargs):
        pass


class Segmenter3DInPatch(Segmenter):

    def __init__(self, mode=None, config=None):
        super(Segmenter3DInPatch, self).__init__()
        self.config = config
        self.ready = False

    def training_setup(self, config):
        pass

    def testing_setup(self, config):
        pass

    def pred_setup(self):
        training_config = load_json_to_dict(self.config["training_config_file"])
        self.partition = Partition(training_config["patch_size"], self.config["overlap_size"],
                                                 padding_mode='reflect', mode="pred")
        # setup model
        self.model = get_network(training_config["model"])(**training_config["model_setting"])
        self.device = torch.device(self.config["device"])
        initialize_model(self.model, ckpoint_path=self.config["ckpoint_path"])
        self.model.to(self.device)

        self.model.eval()
        self.ready = True

    def train(self, model, training_data_loader, validation_data, optimizer, criterion, device, config,
              scheduler=None):
        train_segmentation(model, training_data_loader, validation_data, optimizer, criterion, device, config,
                           scheduler=None)

    def test(self, *args, **kwargs):
        pass

    def segment(self, image):
        pass


class CascadedSegmenter(Segmenter3DInPatch):
    def training_init(self):
        super(CascadedSegmenter, self).training_init()

    def train(self, *args, **kwargs):
        pass

    def test(self, *args, **kwargs):
        pass

    def segment(self, *args, **kwargs):
        pass


class Segmenter3DInPatchClassWise(Segmenter3DInPatch):
    def __init__(self, mode=None, config=None):
        super(Segmenter3DInPatchClassWise, self).__init__(mode, config)
    
    def train(self, *args, **kwargs):
        pass

    def test(self, *args, **kwargs):
        pass

    def segment(self, image, if_output_prob_map=False, if_output_itk=True):
        if not self.ready:
            self.pred_setup()
        sample = {'image': image, 'name': ''}
        image_tiles = self.partition(sample)['image']
        batch_size = self.config['batch_size']
        outputs = []  # a list of lists that store outputs at each step
        # for i in range(1):
        with torch.no_grad():
            for i in range(0, np.ceil(image_tiles.size()[0] / batch_size).astype(int)):
                temp_input = image_tiles.narrow(0, batch_size * i,
                                                batch_size if batch_size * (i + 1) <=
                                                              image_tiles.size()[0]
                                                else image_tiles.size()[0] - batch_size * i)

                # predict through model 1
                temp_output = self.model(temp_input.to(self.device)).cpu()

                outputs.append(temp_output)
                del temp_input

            predictions = torch.sigmoid(torch.cat(outputs, dim=0))

            if not if_output_prob_map:
                predictions = predictions > 0.5

            pred_assemble_FC = self.partition.assemble(predictions[:, 0, :, :, :], if_itk=if_output_itk,
                                                       crop_size=self.config["overlap_size"])
            pred_assemble_TC = self.partition.assemble(predictions[:, 1, :, :, :], if_itk=if_output_itk,
                                                       crop_size=self.config["overlap_size"])

        return pred_assemble_FC, pred_assemble_TC
