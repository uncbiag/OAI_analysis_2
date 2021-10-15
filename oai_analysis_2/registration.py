import easyreg
import demo.demo_for_easyreg_eval as demo
import mermaid

import icon_registration.pretrained_models

import torch


class AVSM_Registration:
    def __init__(self, ckpoint_path, config_path):
        self.ckpoint_path = ckpoint_path
        self.config_path = config_path 


    def register(self, fixed_image, moving_image):


        #write_images

        demo.do_registration_eval(args, [fixed_image_path, moving_image_path, none, none])

class ICON_Registration:
    def __init__(self):
        self.register_object = icon_registration.pretrained_models.OAI_knees_registration_model(pretrained=True)

    def register(self, fixed_image, moving_image):
        self.register_object



