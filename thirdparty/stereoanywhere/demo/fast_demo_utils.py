import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch._dynamo
import tensorrt as trt
import pycuda.driver as cuda

# Initialize CUDA driver inside the main scripts
# This is necessary since the server uses a different cuda context
# import pycuda.autoinit
import time
import tqdm
import os
from torch import autocast


# Custom wrapper for StereoAnywhere (to handle padding and mono model)
class StereoAnywhereWrapper(torch.nn.Module):
    def __init__(self, args, stereo_model, mono_model):
        super(StereoAnywhereWrapper, self).__init__()
        self.args = args
        self.stereo_model = stereo_model
        self.mono_model = mono_model

    def forward(self, left_image, right_image, mono_left=None, mono_right=None):
        # Assuming the model takes a batch of images as input
        if self.mono_model is not None:
            # mono_depths = self.mono_model.infer_image(torch.cat([left_image, right_image], 0), input_size_width=self.args.mono_width, input_size_height=self.args.mono_height)
            mono_depth_left = self.mono_model.infer_image(
                left_image,
                input_size_width=self.args.mono_width,
                input_size_height=self.args.mono_height,
            )
            mono_depth_right = self.mono_model.infer_image(
                right_image,
                input_size_width=self.args.mono_width,
                input_size_height=self.args.mono_height,
            )
            mono_depths = torch.cat([mono_depth_left, mono_depth_right], 0)
            mono_depths = (mono_depths - mono_depths.min()) / (
                mono_depths.max() - mono_depths.min()
            )
            mono_left = mono_depths[0].unsqueeze(0)
            mono_right = mono_depths[1].unsqueeze(0)
        else:
            mono_left = (
                torch.zeros_like(left_image[:, 0:1]) if mono_left is None else mono_left
            )
            mono_right = (
                torch.zeros_like(right_image[:, 0:1])
                if mono_right is None
                else mono_right
            )

        # Pad 32
        ht, wt = left_image.shape[-2], left_image.shape[-1]
        pad_ht = (((ht // 32) + 1) * 32 - ht) % 32
        pad_wd = (((wt // 32) + 1) * 32 - wt) % 32
        _pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]

        left_image = F.pad(left_image, _pad, mode="replicate")
        right_image = F.pad(right_image, _pad, mode="replicate")
        mono_left = F.pad(mono_left, _pad, mode="replicate")
        mono_right = F.pad(mono_right, _pad, mode="replicate")

        pred_disps, _ = self.stereo_model(
            left_image,
            right_image,
            mono_left,
            mono_right,
            test_mode=True,
            iters=self.args.iters,
        )
        pred_disp = -pred_disps.squeeze(1)

        hd, wd = pred_disp.shape[-2:]
        c = [_pad[2], hd - _pad[3], _pad[0], wd - _pad[1]]
        pred_disp = pred_disp[:, c[0] : c[1], c[2] : c[3]]

        return pred_disp


# Depth Anything V2 - TensorRT - Utils


# --------- Load the TensorRT engine ---------
def load_engine(engine_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


# --------- Allocate buffers using named tensors ---------
def dav2_rt_allocate_buffers(context):
    stream = cuda.Stream()

    input_name = "input"
    output_name = "output"

    input_shape = context.get_tensor_shape(input_name)
    output_shape = context.get_tensor_shape(output_name)

    input_dtype = trt.nptype(context.engine.get_tensor_dtype(input_name))
    output_dtype = trt.nptype(context.engine.get_tensor_dtype(output_name))

    input_size = trt.volume(input_shape)
    output_size = trt.volume(output_shape)

    h_input = cuda.pagelocked_empty(input_size, dtype=input_dtype)
    d_input = cuda.mem_alloc(h_input.nbytes)

    h_output = cuda.pagelocked_empty(output_size, dtype=output_dtype)
    d_output = cuda.mem_alloc(h_output.nbytes)

    # Bind addresses
    context.set_tensor_address(input_name, int(d_input))
    context.set_tensor_address(output_name, int(d_output))

    return h_input, d_input, h_output, d_output, stream, input_shape, output_shape


# --------- Preprocess input image ---------
def dav2_rt_preprocess_image(img, input_shape):
    # Assuming a HWC image numpy array
    orig_h, orig_w = img.shape[:2]
    _, _, h, w = input_shape  # assuming NCHW

    # Resize image to the input shape
    img = cv2.resize(img, (w, h))
    # Convert to float32 and normalize
    img = img.astype(np.float32) / 255.0
    # Normalize using ImageNet mean and std
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    for i in range(3):
        img[:, :, i] = (img[:, :, i] - IMAGENET_MEAN[i]) / IMAGENET_STD[i]

    img = img.transpose((2, 0, 1))  # HWC to CHW
    img = np.expand_dims(img, axis=0)  # add batch dim
    return img, (orig_h, orig_w)


# --------- Run inference ---------
def dav2_rt_infer(engine, images):
    if not isinstance(images, list):
        images = [images]

    context = engine.create_execution_context()

    h_input, d_input, h_output, d_output, stream, input_shape, output_shape = (
        dav2_rt_allocate_buffers(context)
    )

    outputs = []

    for image in images:
        # Preprocess and copy to input
        input_image, (orig_h, orig_w) = dav2_rt_preprocess_image(image, input_shape)
        np.copyto(h_input, input_image.ravel())

        # Copy input to device
        cuda.memcpy_htod_async(d_input, h_input, stream)

        # Run inference
        context.execute_async_v3(stream.handle)

        # Copy output from device
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()

        # Post-process
        output = h_output.reshape(output_shape[-2], output_shape[-1])
        output = cv2.resize(output, (orig_w, orig_h))  # Resize to original shape

        outputs.append(output)

    return tuple(outputs) if len(outputs) > 1 else outputs[0]


# Depth Anything V2 - TensorRT - Utils ---- END
