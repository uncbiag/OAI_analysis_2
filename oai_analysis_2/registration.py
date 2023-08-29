import numpy as np
import icon_registration.pretrained_models
import icon_registration.network_wrappers
import icon_registration.itk_wrapper


class AVSM_Registration:
    def __init__(self, ckpoint_path, config_path):
        self.ckpoint_path = ckpoint_path
        self.config_path = config_path

    def register(self, fixed_image, moving_image):
        # Need to install easyreg first
        import demo.demo_for_easyreg_eval as demo
        #write_images
        demo.do_registration_eval(args, [fixed_image_path, moving_image_path, none, none])

class ICON_Registration:
    def __init__(self):
        self.register_module = icon_registration.pretrained_models.OAI_knees_registration_model(pretrained=True)
        icon_registration.network_wrappers.adjust_batch_size(self.register_module, 1)
        self.register_module.cpu()
        #self.register_module.cuda()

    def register(self, fixed_image, moving_image):
        print("fixed range", np.min(fixed_image), np.max(fixed_image))
        print("moving range", np.min(moving_image), np.max(moving_image))
        phi_fixed_moving, phi_moving_fixed = icon_registration.itk_wrapper.register_pair(self.register_module, fixed_image, moving_image)

        return phi_fixed_moving



