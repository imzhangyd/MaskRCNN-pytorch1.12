# MaskRCNN-pytorch1.12
The official code of MaskRCNN https://github.com/facebookresearch/maskrcnn-benchmark works well on old Pytorch. But there are some problems when it is built on Pytorch 1.12. This repository provides code suitable for Pytorch 1.12. You only need to replace these files in the original code with that in this repository.

## Environment
python==3.9.7
torch==1.12.1+cu113
torchvision==0.13.1+cu113
cuda11.3

## How to use
Download the original repository and this repository.
Replace the files in the original repository with the files in this repository.
Then install refer to https://github.com/facebookresearch/maskrcnn-benchmark/blob/main/INSTALL.md.


## Details of change
In setup.py, the line 'cmdclass ...' is revised to cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension.with_options(use_ninja=False)},
Delete: #include <THC/THC.h>
#include <THC/THCDeviceUtils.cuh> --> #include <ATen/cuda/DeviceUtils.cuh>
#include <THC/THCAtomics.cuh> --> #include <ATen/cuda/Atomic.cuh>
THCCeilDiv --> at::ceil_div, and add #include <ATen/ceil_div.h>
Delete the line: THCState *state = at::globalContext().lazyInitCUDA(); // TODO replace with getTHCState
THCudaMalloc --> c10::cuda::CUDACachingAllocator::raw_alloc(size), delete "state"
THCudaFree --> c10::cuda::CUDACachingAllocator::raw_delete(ptr), delete "state"
THCudaCheck --> C10_CUDA_CHECK
AT_CHECK --> TORCH_CHECK

## Reference
https://discuss.pytorch.org/t/question-about-thc-thc-h/147145/8
https://github.com/pytorch/pytorch/issues/72807
https://stackoverflow.com/questions/72988735/replacing-thc-thc-h-module-to-aten-aten-h-module


