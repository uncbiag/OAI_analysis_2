import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import onnxruntime
import itk

class UNet(nn.Module):
    def __init__(self, in_channels, n_classes, bias=False, BN=False):
        self.in_channel = in_channels
        self.n_classes = n_classes
        super(UNet, self).__init__()
        self.ec0 = self.encoder(self.in_channel, 32, bias=bias, batchnorm=BN)
        self.ec1 = self.encoder(32, 64, bias=bias, batchnorm=BN)
        self.ec2 = self.encoder(64, 64, bias=bias, batchnorm=BN)
        self.ec3 = self.encoder(64, 128, bias=bias, batchnorm=BN)
        self.ec4 = self.encoder(128, 128, bias=bias, batchnorm=BN)
        self.ec5 = self.encoder(128, 256, bias=bias, batchnorm=BN)
        self.ec6 = self.encoder(256, 256, bias=bias, batchnorm=BN)
        self.ec7 = self.encoder(256, 512, bias=bias, batchnorm=BN)

        self.pool0 = nn.MaxPool3d(2)
        self.pool1 = nn.MaxPool3d(2)
        self.pool2 = nn.MaxPool3d(2)

        self.dc9 = self.decoder(512, 512, kernel_size=2, stride=2, bias=bias, batchnorm=BN)
        self.dc8 = self.decoder(256 + 512, 256, kernel_size=3, stride=1, padding=1, bias=bias, batchnorm=BN)
        self.dc7 = self.decoder(256, 256, kernel_size=3, stride=1, padding=1, bias=bias, batchnorm=BN)
        self.dc6 = self.decoder(256, 256, kernel_size=2, stride=2, bias=bias, batchnorm=BN)
        self.dc5 = self.decoder(128 + 256, 128, kernel_size=3, stride=1, padding=1, bias=bias, batchnorm=BN)
        self.dc4 = self.decoder(128, 128, kernel_size=3, stride=1, padding=1, bias=bias, batchnorm=BN)
        self.dc3 = self.decoder(128, 128, kernel_size=2, stride=2, bias=bias, batchnorm=BN)
        self.dc2 = self.decoder(64 + 128, 64, kernel_size=3, stride=1, padding=1, bias=bias, batchnorm=BN)
        self.dc1 = self.decoder(64, 64, kernel_size=3, stride=1, padding=1, bias=bias, batchnorm=BN)
        # self.dc0 = self.decoder(64, n_classes, kernel_size=1, stride=1, bias=False)
        self.dc0 = nn.Conv3d(64, n_classes, kernel_size=1, stride=1, padding=0, bias=bias)

        # self.weights_init()


    def weights_init(self):
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                if not m.weight is None:
                    nn.init.xavier_normal_(m.weight.data)
                if not m.bias is None:
                    m.bias.data.zero_()

    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=True, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.ReLU())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.ReLU())
        return layer


    def decoder(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                output_padding=0, bias=True, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                                   padding=padding, output_padding=output_padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.ReLU())
        else:
            layer = nn.Sequential(
                nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                                   padding=padding, output_padding=output_padding, bias=bias),
                nn.ReLU())
        return layer

    def forward(self, x):
        e0 = self.ec0(x)
        syn0 = self.ec1(e0)
        e1 = self.pool0(syn0)
        e2 = self.ec2(e1)
        syn1 = self.ec3(e2)
        del e0, e1, e2

        e3 = self.pool1(syn1)
        e4 = self.ec4(e3)
        syn2 = self.ec5(e4)
        del e3, e4

        e5 = self.pool2(syn2)
        e6 = self.ec6(e5)
        e7 = self.ec7(e6)
        del e5, e6

        d9 = torch.cat((self.dc9(e7), syn2), dim=1)
        del e7, syn2

        d8 = self.dc8(d9)
        d7 = self.dc7(d8)
        del d9, d8

        d6 = torch.cat((self.dc6(d7), syn1), dim=1)
        del d7, syn1

        d5 = self.dc5(d6)
        d4 = self.dc4(d5)
        del d6, d5

        d3 = torch.cat((self.dc3(d4), syn0), dim=1)
        del d4, syn0

        d2 = self.dc2(d3)
        d1 = self.dc1(d2)
        del d3, d2

        d0 = self.dc0(d1)
        return d0

model = UNet(in_channels=1, n_classes=2, bias=True, BN=False)
model.cuda()

model.load_state_dict(torch.load("segmentation_model.pth.tar")["model_state_dict"], strict=True)
model.eval()

dummy_input = torch.randn(1, 1, 128, 128, 32, requires_grad=False)
dummy_input = dummy_input.cuda()

onnx_output_file = 'output_onnx'
torch.onnx.export(model,
                  dummy_input,
                  onnx_output_file,
                  export_params=True,
                  opset_version=11,
                  do_constant_folding=True,
                  input_names = ['input'],
                  output_names = ['output'],
                  dynamic_axes={'input' : {0 : 'batch_size'},
                                'output' : {0 : 'batch_size'}})


def get_inference_session(onnx_filename):
    ''' Generate the onnx inference session.'''
    # TODO change provider to GPU after it is fixed
    return onnxruntime.InferenceSession(
        onnx_filename, providers=['CUDAExecutionProvider'])

device = torch.device("cuda")
#input_batch = np.random.randn(1, 1, 128, 128, 32).astype('float32')

input_batch = itk.imread("/home/pranjal.sahu/OAI_analysis_2/test/test_files/colab_case/image_preprocessed.nii.gz", itk.F)
print('Full image size is ', input_batch.shape)
input_batch = input_batch[:32, :128, :128]
input_batch = np.moveaxis(input_batch, 0, 2)
print(input_batch.shape)
input_batch = np.expand_dims(input_batch, 0)
input_batch = np.expand_dims(input_batch, 0)
input_batch = np.concatenate([input_batch, input_batch, input_batch, input_batch], axis=0)
print(input_batch.shape)

import time

onnx_filename = 'output_onnx.onnx'
ort_session = get_inference_session(onnx_filename)
t1 = time.time()
ort_inputs = {ort_session.get_inputs()[0].name: input_batch}
ort_outs = ort_session.run(None, ort_inputs)[0]
t2 = time.time()
print('ONNX time ', t2 -t1)

model.eval()
t1 = time.time()
input_batch = torch.from_numpy(input_batch).to(device)
pytorch_out = model(input_batch)
pytorch_out = pytorch_out.data.cpu().numpy()
t2 = time.time()
print('Pytorch time ', t2 -t1)

print(pytorch_out.shape)
print(type(pytorch_out))

print('---------------------')
print(ort_outs.shape)
print(type(ort_outs))

np.save('pytorch_out.npy', pytorch_out)
np.save('ort_outs.npy', ort_outs)


c_abs = np.abs(pytorch_out - ort_outs)
print(np.max(c_abs))
print(np.min(c_abs))