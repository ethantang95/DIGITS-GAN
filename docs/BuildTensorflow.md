# Installing TensorFlow

DIGITS now supports TensorFlow as an optional alternative backend to Caffe or Torch.

> NOTE: TensorFlow support is still experimental!

We recommend installing TensorFlow in a fixed and separate environment. This is because TensorFlow support is still in development and stability is not ensured.

Installation for [Ubuntu](https://www.tensorflow.org/install/install_linux#installing_with_virtualenv)

Installation for [Mac](https://www.tensorflow.org/install/install_mac#installing_with_virtualenv)

## Requirements

DIGITS is current targetting tensorflow-gpu V1.1.

TensorFlow for DIGITS requires one or more NVIDIA GPUs with CUDA Compute Capbility of 3.0 or higher. See [the official GPU support list](https://developer.nvidia.com/cuda-gpus) to see if your GPU supports it.

Along with that requirement, the following should be installed

* One or more NVIDIA GPUs ([details](InstallCuda.md#gpu))
* An NVIDIA driver ([details and installation instructions](InstallCuda.md#driver))
* A CUDA toolkit ([details and installation instructions](InstallCuda.md#cuda-toolkit))
* cuDNN ([download page](https://developer.nvidia.com/cudnn))

## Basic Installation

These instructions are based on [the official TensorFlow instructions]
(https://www.tensorflow.org/versions/master/install/)

TensorFlow comes with pip, to install it, just simply use the command
```
pip install tensorflow-gpu=1.1.0
```

TensorFlow should then install effortlessly and pull in all its required dependices.

## Installation From Source

If you would like to build TensorFlow from scratch, the instructions are available on [the official installation guide](https://www.tensorflow.org/versions/master/install/install_sources)

## Getting Started With TensorFlow In DIGITS

Follow [these instructions](GettingStartedTensorFlow.md) for information on getting started with TensorFlow in DIGITS