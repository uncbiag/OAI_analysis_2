import easyreg
import demo.demo_for_easyreg_eval as demo
import mermaid
import numpy as np
import icon_registration.pretrained_models
import icon_registration.network_wrappers
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
        self.register_module = icon_registration.pretrained_models.OAI_knees_registration_model(pretrained=True)
        icon_registration.network_wrappers.adjust_batch_size(self.register_module, 1)
        self.register_module.cuda()

    def register(self, fixed_image, moving_image):

        def preprocess(itk_image):
            
            iA = torch.Tensor(np.array(itk_image))
            iA = torch.nn.functional.avg_pool3d(iA[None, None], 2).cuda()

            print(torch.min(iA), torch.max(iA))

            return iA

        iA = preprocess(fixed_image)
        iB = preprocess(moving_image)
        self.register_module(iA, iB)



