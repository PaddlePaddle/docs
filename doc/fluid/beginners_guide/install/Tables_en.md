-------------------------------------------------------------------------
#APPENDIX

## Compile Dependency Table

|Dependency package name|Version|Description|Installation command|
|:----------:|:--------------:|:------------:|:-----------------:|
|CMake |3.4|                         |
|GCC    |4.8 / 5.4| recommends using devtools2 for CentOS|       |
|Python |2.7.x. |depends on libpython2.7.so |`apt install python-dev` or`yum install python-devel`|
|SWIG   |minimum 2.0|| `apt install swig` or `yum install swig`|
|Wget| any|| `apt install wget` or `yum install wget`|
|Openblas |any|||
|Pip| minimum 9.0.1||`apt install python-pip` or `yum install python-pip`|
|Numpy |>=1.12.0|| `pip install numpy==1.14.0`|
|Protobuf| 3.1.0|| `pip install protobuf==3.1.0`|
|Wheel| any||` pip install wheel`|
|patchELF| any ||`apt install patchelf` or see github [patchELF official documentation](https://gist.github.com/ruario/80fefd174b3395d34c14)|
|Go |>=1.8| optional||
--------------------------------------------------------------------------


## Compile Option Table

Option              | Description                            | Default
--------------------|----------------------------------------|---------
WITH_GPU|Whether to support GPU | ON|
WITH_C_API|Whether to compile API| OFF
WITH_DOUBLE|Whether to use double precision floating point numeber| OFF
WITH_DSO |whether to load CUDA dynamic libraries dynamically at runtime, instead of statically loading CUDA dynamic libraries.|ON
WITH_AVX|whether to compile PaddlePaddle binaries file containing the AVX instruction set |ON
WITH_PYTHON| Whether the PYTHON interpreter is embedded |ON
WITH_STYLE_CHECK| Whether to perform code style checking at compile time |ON
WITH_TESTING| Whether to turn on unit test |OFF
WITH_DOC |Whether to compile Chinese and English documents |OFF
WITH_SWIG_PY|Whether to compile PYTHON's SWIG interface, which can be used for predicting and customizing training |Auto
WITH_GOLANG|Whether to compile the fault-tolerant parameter server of the go language|OFF
WITH_MKL|Whether to use the MKL math library, if not,using OpenBLAS|ON
WITH_SYSTEM_BLAS|Whether to use thesystem's BLAS |OFF 
WITH_DISTRIBUTE|Whether to Compile with distributed version |OFF
WITH_MKL|Whether to uses the MKL math library, if not, using OpenBLAS|ON
WITH_RDMA|Whether to compile the relevant parts that supports RDMA |OFF
WITH_BRPC_RDMA| Whether to use BRPC RDMA as RPC protocol |OFF
ON_INFER |Whether to turn on prediction optimization |OFF
DWITH_ANAKIN |Whether to Compile ANAKIN| OFF

#### BLAS

PaddlePaddle supports two BLAS libraries, [MKL](https://software.intel.com/en-us/mkl) and [OpenBlAS](http://www.openblas.net/). MKL is used by default. If you use MKL and the machine contains the AVX2 instruction set, you will also download the MKL-DNN math library, for details please refer to [here](https://github.com/PaddlePaddle/Paddle/tree/develop/doc/design/mkldnn#cmake).

If you close MKL, OpenBLAS will be used as the BLAS library.

#### CUDA/cuDNN

PaddlePaddle automatically finds the CUDA and cuDNN libraries installed in the system for compilation and execution at compile time/runtime. Use the parameter `-DCUDA_ARCH_NAME=Auto` to specify to enable automatic detection of the SM architecture and speed up compilation.

PaddlePaddle can be compiled and run using any version after cuDNN v5.1, but try to keep the same version of cuDNN compiled and running. We recommend using the latest version of cuDNN.

#### Compile Option Settings

PaddePaddle implements references to various BLAS/CUDA/cuDNN libraries by specifying paths at compile time. When cmake compiles, it first searches the system paths ( `/usr/liby` and `/usr/local/lib` ) for these libraries, and also reads the relevant path variables for searching. Can be set by using the -D command, for example:

>`Cmake .. -DWITH_GPU=ON -DWITH_TESTING=OFF -DCUDNN_ROOT=/opt/cudnnv5`

**Note**: The settings for these compilation options are only valid for the first cmake. If you want to reset it later, it is recommended to clean up the entire build directory ( rm -rf ) and then specify it.

---------------------------------------------------------------------------

## Installation Package List

Version number |release notes
---------------|-------------------
Paddlepaddle==[version number] such as paddlepaddle==1.0.1 (download version 1.0.1 only supports CPU PaddlePaddle) |Only support the corresponding version of the CPU PaddlePaddle, please refer to [Pypi](https://pypi.org/project/paddlepaddle/#history) for the specific version.
Paddlepaddle-gpu==1.0.1| Using version 1.0.1 compiled with CUDA 9.0 and cuDNN 7
Paddlepaddle-gpu==1.0.1.post87|Using version 1.0.1 compiled with CUDA 8.0 and cuDNN 7
Paddlepaddle-gpu==1.0.1.post85|Using version 1.0.1 compiled with CUDA 8.0 and cuDNN 5
Paddlepaddle-gpu==1.0.0 |Using version 1.0.0 compiled with CUDA 9.0 and cuDNN 7
Paddlepaddle-gpu==1.0.0.post87|Using version 1.0.0 compiled with CUDA 8.0 and cuDNN 7
Paddlepaddle-gpu==1.0.0.post85|Using version 1.0.0 compiled with CUDA 8.0 and cuDNN 5
Paddlepaddle-gpu==0.15.0 |Using version 0.15.0 compiled with CUDA 9.0 and cuDNN 7
Paddlepaddle-gpu==0.15.0.post87|Using version 0.15.0 compiled with CUDA 8.0 and cuDNN 7
Paddlepaddle-gpu==0.15.0.post85|Using version 0.15.0 compiled with CUDA 8.0 and cuDNN 5
Paddlepaddle-gpu==0.14.0 |Using version 0.15.0 compiled with CUDA 9.0 and cuDNN 7
Paddlepaddle-gpu==0.14.0.post87|Using version 0.15.0 compiled with CUDA 8.0 and cuDNN 7
Paddlepaddle-gpu==0.14.0.post85|Using version 0.15.0 compiled with CUDA 8.0 and cuDNN 5
Paddlepaddle-gpu==0.13.0 |Using version 0.13.0 compiled with CUDA 9.0 and cuDNN 7
Paddlepaddle-gpu==0.12.0 |Using version 0.12.0 compiled with CUDA 8.0 and cuDNN 5
Paddlepaddle-gpu==0.11.0.post87|Using version 0.11.0 compiled with CUDA 8.0 and cuDNN 7
Paddlepaddle-gpu==0.11.0.post85 |Using version 0.11.0 compiled with CUDA 8.0 and cuDNN 5
Paddlepaddle-gpu==0.11.0|Using version 0.11.0 compiled with CUDA 7.5 and cuDNN 5

You can find the various distributions of PaddlePaddle-gpu in [the Release History](https://pypi.org/project/paddlepaddle-gpu/#history).

-----------------------------------------------------------------------

## Installation Mirror Table and Introduction

Version number    | release instruction
------------------|------------------------
Hub.baidubce.com/paddlepaddle/paddle:latest |The latest pre-installed image of the PaddlePaddle CPU version
Hub.baidubce.com/paddlepaddle/paddle:latest-dev |The latest PaddlePaddle development environment
Hub.baidubce.com/paddlepaddle/paddle:[Version]| Replace version with a specific version, pre-installed PaddlePaddle image in historical version
Hub.baidubce.com/paddlepaddle/paddle:latest-gpu |The latest pre-installed image of the PaddlePaddle GPU version

You can find the docker image for each release of PaddlePaddle in the [DockerHub](https://hub.docker.com/r/paddlepaddle/paddle/tags/).

-----------------------------------------------------------------------


## Multi-version whl package list-Release
Release Instruction| cp27-cp27mu|cp27-cp27m| cp35-cp35m| cp36-cp36m| cp37-cp37m
---------|----------|---------|----------|----------|-----------|
Cpu-noavx-mkl|[paddlepaddle-1.2.0-cp27-cp27mu-linux_x86_64.whl](http://paddlepaddle.org/download?url=http://paddle-wheel.bj.bcebos.com/1.2.0-cpu-noavx-mkl/paddlepaddle-1.2.0-cp27-cp27mu-linux_x86_64.whl) |[paddlepaddle-1.2.0-cp27-cp27m-linux_x86_64.whl](http://paddlepaddle.org/download?url=http://paddle-wheel.bj.bcebos.com/1.2.0-cpu-noavx-mkl/paddlepaddle-1.2.0-cp27-cp27m-linux_x86_64.whl)|[paddlepaddle-1.2.0-cp35-cp35m-linux_x86_64.whl](http://paddlepaddle.org/download?url=http://paddle-wheel.bj.bcebos.com/1.2.0-cpu-noavx-mkl/paddlepaddle-1.2.0-cp35-cp35m-linux_x86_64.whl)| [paddlepaddle-1.2.0- Cp36-cp36m-linux_x86_64.whl ](http://paddlepaddle.org/download?url=http://paddle-wheel.bj.bcebos.com/1.2.0-cpu-noavx-mkl/paddlepaddle-1.2.0-cp36-cp36m-linux_x86_64.whl)|[paddlepaddle-1.2.0-cp37-cp37m-linux_x86_64.whl](http://paddlepaddle.org/download?url=http://paddle-wheel.bj.bcebos.com/1.2.0-cpu-noavx-mkl/paddlepaddle-1.2.0-cp37-cp37m-linux_x86_64.whl)
Cpu_avx_mkl| [paddlepaddle-1.2.0-cp27-cp27mu-linux_x86_64.whl](http://paddlepaddle.org/download?url=http://paddle-wheel.bj.bcebos.com/1.2.0-cpu-avx-mkl/paddlepaddle-1.2.0-cp27-cp27mu-linux_x86_64.whl)| [paddlepaddle-1.2.0-cp27-cp27m-linux_x86_64.whl](http://paddle-wheel.bj.bcebos.com/1.2.0-cpu-avx-mkl/paddlepaddle-1.2.0-cp27-cp27m-linux_x86_64.whl) |[paddlepaddle-1.2.0-cp35-cp35m-linux_x86_64.whl](http://paddle-wheel.bj.bcebos.com/1.2.0-cpu-avx-mkl/paddlepaddle-1.2.0-cp35-cp35m-linux_x86_64.whl) | [paddlepaddle-1.2.0-cp36-cp36m- Linux_x86_64.whl](http://paddle-wheel.bj.bcebos.com/1.2.0-cpu-avx-mkl/paddlepaddle-1.2.0-cp36-cp36m-linux_x86_64.whl) |[paddlepaddle-1.2.0-cp37-cp37m-linux_x86_64.whl](http://paddle-wheel.bj.bcebos.com/1.2.0-cpu-avx-mkl/paddlepaddle-1.2.0-cp37-cp37m-linux_x86_64.whl)
Cpu_avx_openblas|[paddlepaddle-1.2.0-cp27-cp27mu-linux_x86_64.whl](http://paddle-wheel.bj.bcebos.com/1.2.0-cpu-avx-openblas/paddlepaddle-1.2.0-cp27-cp27mu-linux_x86_64.whl) |[paddlepaddle-1.2.0-cp27-cp27m-linux_x86_64.whl](http://paddlepaddle.org/download?url=http://paddle-wheel.bj.bcebos.com/1.2.0-cpu-avx-openblas/paddlepaddle-1.2.0-cp27-cp27m-linux_x86_64.whl) |[paddlepaddle-1.2.0-cp35-cp35m-linux_x86_64.whl](http://paddlepaddle.org/download?url=http://paddle-wheel.bj.bcebos.com/1.2.0-cpu-avx-openblas/paddlepaddle-1.2.0-cp35-cp35m-linux_x86_64.whl)| [paddlepaddle-1.2.0-cp36-cp36m- Linux_x86_64.whl](http://paddlepaddle.org/download?url=http://paddle-wheel.bj.bcebos.com/1.2.0-cpu-avx-openblas/paddlepaddle-1.2.0-cp36-cp36m-linux_x86_64.whl) |[paddlepaddle-1.2.0-cp37-cp37m-linux_x86_64.whl](http://paddle-wheel.bj.bcebos.com/1.2.0-cpu-avx-openblas/paddlepaddle-1.2.0-cp37-cp37m-linux_x86_64.whl)
Cuda8.0_cudnn5_avx_mkl|[paddlepaddle_gpu-1.2.0-cp27-cp27mu-linux_x86_64.whl](http://paddle-wheel.bj.bcebos.com/1.2.0-gpu-cuda8-cudnn5-avx-mkl/paddlepaddle_gpu-1.2.0.post85-cp27-cp27mu-linux_x86_64.whl) |[paddlepaddle_gpu-1.2.0-cp27-cp27m-linux_x86_64.whl](http://paddle-wheel.bj.bcebos.com/1.2.0-gpu-cuda8-cudnn5-avx-mkl/paddlepaddle_gpu-1.2.0.post85-cp27-cp27m-linux_x86_64.whl)| [paddlepaddle_gpu-1.2.0-cp35-cp35m-linux_x86_64.whl](http://paddle-wheel.bj.bcebos.com/1.2.0-gpu-cuda8-cudnn5-avx-mkl/paddlepaddle_gpu-1.2.0.post85-cp35-cp35m-linux_x86_64.whl)|[ paddlepaddle_gpu-1.2.0-cp36- Cp36m-linux_x86_64.whl](http://paddle-wheel.bj.bcebos.com/1.2.0-gpu-cuda8-cudnn5-avx-mkl/paddlepaddle_gpu-1.2.0.post85-cp36-cp36m-linux_x86_64.whl)| [paddlepaddle_gpu-1.2.0-cp37-cp37m-linux_x86_64.whl](http://paddle-wheel.bj.bcebos.com/1.2.0-gpu-cuda8-cudnn5-avx-mkl/paddlepaddle_gpu-1.2.0.post85-cp37-cp37m-linux_x86_64.whl)
Cuda8.0_cudnn7_noavx_mkl| [paddlepaddle_gpu-1.2.0-cp27-cp27mu-linux_x86_64.whl](http://paddle-wheel.bj.bcebos.com/1.2.0-gpu-cuda8-cudnn7-noavx-mkl/paddlepaddle_gpu-1.2.0-cp27-cp27mu-linux_x86_64.whl)|[paddlepaddle_gpu-1.2.0-cp27-cp27m-linux_x86_64.whl](http://paddle-wheel.bj.bcebos.com/1.2.0-gpu-cuda8-cudnn7-noavx-mkl/paddlepaddle_gpu-1.2.0-cp27-cp27m-linux_x86_64.whl) |[paddlepaddle_gpu-1.2.0-cp35-cp35m-linux_x86_64.whl](http://paddle-wheel.bj.bcebos.com/1.2.0-gpu-cuda8-cudnn7-noavx-mkl/paddlepaddle_gpu-1.2.0-cp35-cp35m-linux_x86_64.whl)| [paddlepaddle_gpu-1.2.0-cp36-Cp36m-linux_x86_64.whl](http://paddle-wheel.bj.bcebos.com/1.2.0-gpu-cuda8-cudnn7-noavx-mkl/paddlepaddle_gpu-1.2.0-cp36-cp36m-linux_x86_64.whl)| [paddlepaddle_gpu-1.2.0-cp37-cp37m-linux_x86_64.whl](http://paddle-wheel.bj.bcebos.com/1.2.0-gpu-cuda8-cudnn7-noavx-mkl/paddlepaddle_gpu-1.2.0-cp37-cp37m-linux_x86_64.whl)
Cuda8.0_cudnn7_avx_mkl| [paddlepaddle_gpu-1.2.0.post87-cp27-cp27mu-linux_x86_64.whl](http://paddle-wheel.bj.bcebos.com/1.2.0-gpu-cuda8-cudnn7-avx-mkl/paddlepaddle_gpu-1.2.0.post87-cp27-cp27mu-linux_x86_64.whl)| [paddlepaddle_gpu-1.2.0.post87-cp27-cp27m-linux_x86_64.whl](http://paddle-wheel.bj.bcebos.com/1.2.0-gpu-cuda8-cudnn7-avx-mkl/paddlepaddle_gpu-1.2.0.post87-cp27-cp27m-linux_x86_64.whl)| [paddlepaddle_gpu-1.2.0.post87-cp35-cp35m-linux_x86_64.whl](http://paddle-wheel.bj.bcebos.com/1.2.0-gpu-cuda8-cudnn7-avx-mkl/paddlepaddle_gpu-1.2.0.post87-cp35-cp35m-linux_x86_64.whl) |[paddlepaddle_gpu- 1.2.0.post87-cp36-cp36m-linux_x86_64.whl](http://paddle-wheel.bj.bcebos.com/1.2.0-gpu-cuda8-cudnn7-avx-mkl/paddlepaddle_gpu-1.2.0.post87-cp36-cp36m-linux_x86_64.whl) |[paddlepaddle_gpu-1.2.0.post87-cp36-cp36m-linux_x86_64.whl](http://paddle-wheel.bj.bcebos.com/1.2.0-gpu-cuda8-cudnn7-avx-mkl/paddlepaddle_gpu-1.2.0.post87-cp37-cp37m-linux_x86_64.whl)
Cuda9.0_cudnn7_avx_mkl| [paddlepaddle_gpu-1.2.0-cp27-cp27mu-linux_x86_64.whl](http://paddle-wheel.bj.bcebos.com/1.2.0-gpu-cuda9-cudnn7-avx-mkl/paddlepaddle_gpu-1.2.0.post97-cp27-cp27mu-linux_x86_64.whl)| [paddlepaddle_gpu-1.2.0-cp27-cp27m-linux_x86_64.whl](http://paddle-wheel.bj.bcebos.com/1.2.0-gpu-cuda9-cudnn7-avx-mkl/paddlepaddle_gpu-1.2.0.post97-cp27-cp27m-linux_x86_64.whl)| [paddlepaddle_gpu-1.2.0-cp35-cp35m-linux_x86_64.whl](http://paddle-wheel.bj.bcebos.com/1.2.0-gpu-cuda9-cudnn7-avx-mkl/paddlepaddle_gpu-1.2.0.post97-cp35-cp35m-linux_x86_64.whl) |[paddlepaddle_gpu-1.2.0-cp36- Cp36m-linux_x86_64.whl](http://paddle-wheel.bj.bcebos.com/1.2.0-gpu-cuda9-cudnn7-avx-mkl/paddlepaddle_gpu-1.2.0.post97-cp36-cp36m-linux_x86_64.whl)| [paddlepaddle_gpu-1.2.0-cp37-cp37m-linux_x86_64.whl](http://paddle-wheel.bj.bcebos.com/1.2.0-gpu-cuda9-cudnn7-avx-mkl/paddlepaddle_gpu-1.2.0.post97-cp37-cp37m-linux_x86_64.whl)



## Multi-version whl package list - dev

Release Instruction| cp27-cp27mu| cp27-cp27m| cp35-cp35m |cp36-cp36m |cp37-cp37m
--------|------------|-----------|------------|-------------|-------------
Cpu-noavx-mkl| [paddlepaddle-latest-cp27-cp27mu-linux_x86_64.whl](http://paddle-wheel.bj.bcebos.com/latest-cpu-noavx-mkl/paddlepaddle-latest-cp27-cp27mu-linux_x86_64.whl)| [paddlepaddle-latest-cp27-cp27m-linux_x86_64.whl](http://paddle-wheel.bj.bcebos.com/latest-cpu-noavx-mkl/paddlepaddle-latest-cp27-cp27m-linux_x86_64.whl)|[paddlepaddle-latest-cp35-cp35m-linux_x86_64.whl](http://paddle-wheel.bj.bcebos.com/latest-cpu-noavx-mkl/paddlepaddle-latest-cp35-cp35m-linux_x86_64.whl)|[ paddlepaddle-latest-cp36-cp36m-linux_x86_64.whl](http://paddle-wheel.bj.bcebos.com/latest-cpu-noavx-mkl/paddlepaddle-latest-cp36-cp36m-linux_x86_64.whl) |[paddlepaddle -latest-cp37-cp37m-linux_x86_64.whl](http://paddle-wheel.bj.bcebos.com/latest-cpu-noavx-mkl/paddlepaddle-latest-cp37-cp37m-linux_x86_64.whl)
Cpu_avx_mkl| [paddlepaddle-latest-cp27-cp27mu-linux_x86_64.whl](http://paddle-wheel.bj.bcebos.com/latest-cpu-avx-mkl/paddlepaddle-latest-cp27-cp27mu-linux_x86_64.whl)|[paddlepaddle-latest-cp27-cp27m-linux_x86_64.whl](http://paddle-wheel.bj.bcebos.com/latest-cpu-avx-mkl/paddlepaddle-latest-cp27-cp27m-linux_x86_64.whl)| [paddlepaddle-latest-cp35-cp35m-linux_x86_64.whl](http://paddle-wheel.bj.bcebos.com/latest-cpu-avx-mkl/paddlepaddle-latest-cp35-cp35m-linux_x86_64.whl)| [paddlepaddle-latest-cp36-cp36m-linux_x86_64.whl](http://paddle-wheel.bj.bcebos.com/latest-cpu-avx-mkl/paddlepaddle-latest-cp36-cp36m-linux_x86_64.whl) |[paddlepaddle-latest-cp37 -cp37m-linux_x86_64.whl](http://paddle-wheel.bj.bcebos.com/latest-cpu-avx-mkl/paddlepaddle-latest-cp37-cp37m-linux_x86_64.whl)
cpu_avx_openblas| [paddlepaddle-latest-cp27-cp27mu-linux_x86_64.whl](http://paddle-wheel.bj.bcebos.com/latest-cpu-avx-openblas/paddlepaddle-latest-cp27-cp27mu-linux_x86_64.whl) |[paddlepaddle-latest-cp27-cp27m-linux_x86_64.whl](http://paddle-wheel.bj.bcebos.com/latest-cpu-avx-openblas/paddlepaddle-latest-cp27-cp27m-linux_x86_64.whl)| [paddlepaddle-latest-cp35-cp35m-linux_x86_64.whl](http://paddle-wheel.bj.bcebos.com/latest-cpu-avx-openblas/paddlepaddle-latest-cp35-cp35m-linux_x86_64.whl)| [paddlepaddle-latest-cp36-cp36m-linux_x86_64.whl](http://paddle-wheel.bj.bcebos.com/latest-cpu-avx-openblas/paddlepaddle-latest-cp36-cp36m-linux_x86_64.whl)| [paddlepaddle-latest-cp37 -cp37m-linux_x86_64.whl](http://paddle-wheel.bj.bcebos.com/latest-cpu-avx-openblas/paddlepaddle-latest-cp37-cp37m-linux_x86_64.whl)
cuda8.0_cudnn5_avx_mkl |[paddlepaddle_gpu-latest-cp27-cp27mu-linux_x86_64.whl](http://paddle-wheel.bj.bcebos.com/latest-gpu-cuda8-cudnn5-avx-mkl/paddlepaddle_gpu-latest-cp27-cp27mu-linux_x86_64.whl)| [paddlepaddle_gpu-latest-cp27-cp27m-linux_x86_64.whl](http://paddle-wheel.bj.bcebos.com/latest-gpu-cuda8-cudnn5-avx-mkl/paddlepaddle_gpu-latest-cp27-cp27m-linux_x86_64.whl) |[paddlepaddle_gpu-latest-cp35-cp35m-linux_x86_64.whl](http://paddle-wheel.bj.bcebos.com/latest-gpu-cuda8-cudnn5-avx-mkl/paddlepaddle_gpu-latest-cp35-cp35m-linux_x86_64.whl)| [paddlepaddle_gpu-latest-cp36-cp36m-linux_x86_64.whl](http://paddle-wheel.bj.bcebos.com/latest-gpu-cuda8-cudnn5-avx-mkl/paddlepaddle_gpu-latest-cp36-cp36m-linux_x86_64.whl)| [paddlepaddle_gpu-latest -cp37-cp37m-linux_x86_64.whl](http://paddle-wheel.bj.bcebos.com/latest-gpu-cuda8-cudnn5-avx-mkl/paddlepaddle_gpu-latest-cp37-cp37m-linux_x86_64.whl)
cuda8.0_cudnn7_noavx_mkl| [paddlepaddle_gpu-latest-cp27-cp27mu-linux_x86_64.whl](http://paddle-wheel.bj.bcebos.com/latest-gpu-cuda8-cudnn7-noavx-mkl/paddlepaddle_gpu-latest-cp27-cp27mu-linux_x86_64.whl) |[paddlepaddle_gpu-latest-cp27-cp27m-linux_x86_64.whl](http://paddle-wheel.bj.bcebos.com/latest-gpu-cuda8-cudnn7-noavx-mkl/paddlepaddle_gpu-latest-cp27-cp27m-linux_x86_64.whl) |[paddlepaddle_gpu-latest-cp35-cp35m-linux_x86_64.whl](http://paddle-wheel.bj.bcebos.com/latest-gpu-cuda8-cudnn7-noavx-mkl/paddlepaddle_gpu-latest-cp35-cp35m-linux_x86_64.whl)| [paddlepaddle_gpu-latest-cp36-cp36m-linux_x86_64.whl](http://paddle-wheel.bj.bcebos.com/latest-gpu-cuda8-cudnn7-noavx-mkl/paddlepaddle_gpu-latest-cp36-cp36m-linux_x86_64.whl)| [paddlepaddle_gpu-latest -cp37-cp37m-linux_x86_64.whl](http://paddle-wheel.bj.bcebos.com/latest-gpu-cuda8-cudnn7-noavx-mkl/paddlepaddle_gpu-latest-cp37-cp37m-linux_x86_64.whl)
cuda8.0_cudnn7_avx_mkl|[paddlepaddle_gpu-latest-cp27-cp27mu-linux_x86_64.whl](http://paddle-wheel.bj.bcebos.com/latest-gpu-cuda8-cudnn7-avx-mkl/paddlepaddle_gpu-latest-cp27-cp27mu-linux_x86_64.whl)| [paddlepaddle_gpu-latest-cp27-cp27m-linux_x86_64.whl](http://paddle-wheel.bj.bcebos.com/latest-gpu-cuda8-cudnn7-avx-mkl/paddlepaddle_gpu-latest-cp27-cp27m-linux_x86_64.whl) |[paddlepaddle_gpu-latest-cp35-cp35m-linux_x86_64.whl](http://paddle-wheel.bj.bcebos.com/latest-gpu-cuda8-cudnn7-avx-mkl/paddlepaddle_gpu-latest-cp35-cp35m-linux_x86_64.whl)| [paddlepaddle_gpu-latest-cp36-cp36m-linux_x86_64.whl](http://paddle-wheel.bj.bcebos.com/latest-gpu-cuda8-cudnn7-avx-mkl/paddlepaddle_gpu-latest-cp36-cp36m-linux_x86_64.whl)|[paddlepaddle_gpu-latest -cp37-cp37m-linux_x86_64.whl](http://paddle-wheel.bj.bcebos.com/latest-gpu-cuda8-cudnn7-avx-mkl/paddlepaddle_gpu-latest-cp37-cp37m-linux_x86_64.whl)
cuda9.0_cudnn7_avx_mkl| [paddlepaddle_gpu-latest-cp27-cp27mu-linux_x86_64.whl](http://paddle-wheel.bj.bcebos.com/latest-gpu-cuda9-cudnn7-avx-mkl/paddlepaddle_gpu-latest-cp27-cp27mu-linux_x86_64.whl)| [paddlepaddle_gpu-latest-cp27-cp27m-linux_x86_64.whl](http://paddle-wheel.bj.bcebos.com/latest-gpu-cuda9-cudnn7-avx-mkl/paddlepaddle_gpu-latest-cp27-cp27m-linux_x86_64.whl) |[paddlepaddle_gpu-latest-cp35-cp35m-linux_x86_64.whl](http://paddle-wheel.bj.bcebos.com/latest-gpu-cuda9-cudnn7-avx-mkl/paddlepaddle_gpu-latest-cp35-cp35m-linux_x86_64.whl)| [paddlepaddle_gpu-latest-cp36-cp36m-linux_x86_64.whl](http://paddle-wheel.bj.bcebos.com/latest-gpu-cuda9-cudnn7-avx-mkl/paddlepaddle_gpu-latest-cp36-cp36m-linux_x86_64.whl) |[paddlepaddle_gpu-latest -cp37-cp37m-linux_x86_64.whl](http://paddle-wheel.bj.bcebos.com/latest-gpu-cuda9-cudnn7-avx-mkl/paddlepaddle_gpu-latest-cp37-cp37m-linux_x86_64.whl)



## Execute the PaddlePaddle training program in Docker
----------------------------------------------------------------------------

Suppose you have written a PaddlePaddle program in the current directory (such as /home/work): `train.py` (can refer to PaddlePaddleBook to write), you can start the training with the following command:

```

Cd /home/work
Docker run -it -v $PWD:/work hub.baidubce.com/paddlepaddle/paddle /work/train.py

```

In the above commands, the `-it` parameter indicates that the container has been run interactively; `-v $PWD:/work` specifies that the current path (the absolute path where the PWD variable in Linux will expand to the current path) is mounted to the `:/work` directory inside the container: `Hub.baidubce.com/paddlepaddle/paddle` specifies the container to be used; finally `/work/train.py` is the command executed inside the container, ie running the training program.

Of course, you can also go into the Docker container and execute or debug your code interactively:

```

Docker run -it -v $PWD:/work hub.baidubce.com/paddlepaddle/paddle /bin/bash
Cd /work
Python train.py

```

**Note: PaddlePaddle Docker image In order to reduce the size, vim is not installed by default. You can edit the code in the container after executing `apt-get install -y vim` in the container.**


## Start PaddlePaddle Book tutorial with Docker
-------------------------------------------------------------------------

Use Docker to quickly launch a local Jupyter Notebook containing the PaddlePaddle official Book tutorial, which can be viewed on the web. PaddlePaddle Book is an interactive Jupyter Notebook for users and developers. If you want to learn more about deep learning, PaddlePaddle Book is definitely your best choice. You can read tutorials or create and share interactive documents with code, formulas, charts, and text.

We provide a Docker image that can run the PaddlePaddle Book directly, running directly:

`Docker run -p 8888:8888 hub.baidubce.com/paddlepaddle/book`

Domestic users can use the following image source to speed up access:

`Docker run -p 8888:8888 hub.baidubce.com/paddlepaddle/book`

Then enter the following URL in your browser:

`Http://localhost:8888/`

It's that simple and enjoy your journey! Please refer to the [FAQ](http://paddlepaddle.org/documentation/docs/zh/1.2/beginners_guide/install/Tables.html#FAQ) if you have any other questions.



## Perform GPU training using Docker
---------------------------------------------------------------------------- 

In order to ensure that the GPU driver works properly in the image, we recommend using [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) to run the image. Don't forget to install the latest GPU drivers on your physical machine in advance.

`Nvidia-docker run -it -v $PWD:/work hub.baidubce.com/paddlepaddle/paddle:latest-gpu /bin/bash`

**Note: If you don't have nvidia-docker installed, you can try the following to mount the CUDA library and Linux devices into the Docker container:**

```

	Export CUDA_SO="$(\ls /usr/lib64/libcuda* | xargs -I{} echo '-v {}:{}') \
	$(\ls /usr/lib64/libnvidia* | xargs -I{} echo '-v {}:{}')"
	Export DEVICES=$(\ls /dev/nvidia* | xargs -I{} echo '--device {}:{}')
	Docker run ${CUDA_SO} \
 	 ${DEVICES} -it hub.baidubce.com/paddlepaddle/paddle:latest-gpu

```

**About AVX:**

AVX is a set of CPU instructions that speed up the calculation of PaddlePaddle. The latest PaddlePaddle Docker image is enabled by default for AVX compilation, so if your computer does not support AVX, you need to [compile](http://paddlepaddle.org/build_from_source_cn.html) PaddlePaddle to no-avx version separately.

The following instructions can check if the Linux computer supports AVX:

`If cat /proc/cpuinfo | grep -i avx; then echo Yes; else echo No; fi`

If the output is No, you need to choose a mirror that uses no-AVX.