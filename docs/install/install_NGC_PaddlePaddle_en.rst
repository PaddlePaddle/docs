..  _install_NGC_PaddlePaddle_container introduction:

==============================================
NVIDIA PaddlePaddle Container Installation Guide
==============================================

----------------------
  Overview
----------------------

The PaddlePaddle NGC Container is optimized for GPU acceleration, and contains a validated set of libraries that enable and optimize GPU performance. This container may also contain modifications to the PaddlePaddle source code in order to maximize performance and compatibility. This container also contains software for accelerating ETL (`DALI <https://developer.nvidia.com/dali/>`_, `RAPIDS <https://rapids.ai/>`_), Training(`cuDNN <https://developer.nvidia.com/cudnn>`_, `NCCL <https://developer.nvidia.com/nccl>`_), and Inference(`TensorRT <https://docs.nvidia.com/deeplearning/frameworks/tf-trt-user-guide/index.html>`_) workloads。

------------------------------------------
  Environmental preparation
------------------------------------------

* Need to be run in a Linux OS environment

Using the PaddlePaddle NGC Container requires the host system to have the following installed:

* `Docker Engine <https://docs.docker.com/get-docker/>`_

* `NVIDIA GPU Drivers <https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html>`_

* `NVIDIA Container Toolkit <https://github.com/NVIDIA/nvidia-docker>`_

For supported versions, see the `Framework Containers Support Matrix <https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html>`_ and the `NVIDIA Container Toolkit Documentation <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html>`_ .

No other installation, compilation, or dependency management is required. It is not necessary to install the NVIDIA CUDA Toolkit.

----------------------
  Installation
----------------------

To run a container, issue the appropriate command as explained in the `Running A Container <https://docs.nvidia.com/deeplearning/frameworks/user-guide/index.html#runcont>`_ chapter in the NVIDIA Containers For Deep Learning Frameworks User’s Guide and specify the registry, repository, and tags. For more information about using NGC, refer to the `NGC Container User Guide <https://docs.nvidia.com/ngc/ngc-catalog-user-guide/index.html>`_ .
If you have Docker 19.03 or later, a typical command to launch the container is:


    ::

        docker run --gpus all --shm-size=1g --ulimit memlock=-1 -it --rm nvcr.io/nvidia/paddlepaddle:yy.mm-py3


If you have Docker 19.02 or earlier, a typical command to launch the container is:


    ::

        nvidia-docker run --shm-size=1g --ulimit memlock=-1 -it --rm nvcr.io/nvidia/paddlepaddle:yy.mm-py3



Where:

* yy.mm is the container version, e.g., the version published in Sept 2022 is 22.09.

PaddlePaddle is run by importing it as a Python module:


    ::

        $ python -c 'import paddle; paddle.utils.run_check()'
       Running verify PaddlePaddle program ...
       W0516 06:36:54.208734   442 device_context.cc:451] Please NOTE: device: 0, GPU Compute Capability: 8.0, Driver API Version: 11.7, Runtime API Version: 11.7
       W0516 06:36:54.212574   442 device_context.cc:469] device: 0, cuDNN Version: 8.4.
       PaddlePaddle works well on 1 GPU.
       W0516 06:37:12.706600   442 fuse_all_reduce_op_pass.cc:76] Find all_reduce operators: 2. To make the speed faster, some all_reduce ops are fused during training, after fusion, the number of all_reduce ops is 2.
       PaddlePaddle works well on 8 GPUs.
       PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now.

See /workspace/README.md inside the container for information on getting started and customizing your PaddlePaddle image.

You might want to pull in data and model descriptions from locations outside the container for use by PaddlePaddle. To accomplish this, the easiest method is to mount one or more host directories as `Docker bind mounts <https://docs.docker.com/storage/bind-mounts/>`_. For example:

    ::

        docker run --gpus all -it --rm -v local_dir:container_dir nvcr.io/nvidia/paddlepaddle:22.07-py3


Note: In order to share data between ranks, NCCL may require shared system memory for IPC and pinned (page-locked) system memory resources. The operating system's limits on these resources may need to be increased accordingly. Refer to your system's documentation for details. In particular, Docker containers default to limited shared and pinned memory resources. When using NCCL inside a container, it is recommended that you increase these resources by issuing:


    ::

        --shm-size=1g --ulimit memlock=-1

in the docker run command.


------------------------------------------
  Introduction to NGC Container
------------------------------------------

For the full list of contents, see the `NGC PaddlePaddle Container Release Notes <https://docs.nvidia.com/deeplearning/frameworks/paddle-paddle-release-notes/index.html>`_.

This container image contains the complete source of the NVIDIA version of PaddlePaddle in /opt/paddle/paddle. It is prebuilt and installed as a system Python module. Visit paddlepaddle.org.cn to learn more about PaddlePaddle.

The NVIDIA PaddlePaddle Container is optimized for use with NVIDIA GPUs, and contains the following software for GPU acceleration:


* `CUDA <https://developer.nvidia.com/cuda-toolkit>`_

* `cuBLAS <https://developer.nvidia.com/cublas>`_

* `NVIDIA cuDNN <https://developer.nvidia.com/cudnn>`_

* `NVIDIA NCCL <https://developer.nvidia.com/nccl>`_ (optimized for `NVLink <http://www.nvidia.com/object/nvlink.html>`_ )

* `NVIDIA Data Loading Library (DALI) <https://developer.nvidia.com/dali>`_

* `TensorRT <https://developer.nvidia.com/tensorrt>`__

* `PaddlePaddle with TensorRT (Paddle-TRT) <https://github.com/PaddlePaddle/Paddle-Inference-Demo/blob/master/docs/optimize/paddle_trt_en.rst>`_

The software stack in this container has been validated for compatibility, and does not require any additional installation or compilation from the end user. This container can help accelerate your deep learning workflow from end to end.


--------------------------------------------
  License
--------------------------------------------

By pulling and using the container, you accept the terms and conditions of this `End User License Agreement <https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license>`_ .
