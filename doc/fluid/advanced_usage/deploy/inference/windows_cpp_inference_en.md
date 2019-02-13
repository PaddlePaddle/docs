Model Inference on Windows
===========================

Set up Environment
-------------------

### Hardware Environment

Hardware Configuration of the experimental environment:

| CPU           |      I7-8700K      |
|:--------------|:-------------------|
| Memory        | 16G               |
| Hard Disk     | 1T hdd + 256G ssd |
| Graphics Card | GTX1080 8G        |

The operating system is win10 family version in the experimental environment.

### Steps to Configure Environment

**Please strictly follow the subsequent steps to install, otherwise the installation may fail**

**Install Visual Studio 2015 update3**

Install Visual Studio 2015. Please choose "customize" for the options of contents to be installed and choose to install all functions relevant to c, c++ and vc++.


Usage of Inference demo
------------------------

Decompress Paddle, Release and fluid_install_dir compressed package.

First enter into Paddle/paddle/fluid/inference/api/demo_ci, then create and enter into directory /build, finally use cmake to generate vs2015 solution file.
Commands are as follows:

`cmake .. -G "Visual Studio 14 2015 Win64" -DWITH_GPU=OFF -DWITH_MKL=OFF -DWITH_STATIC_LIB=ON -DCMAKE_BUILD_TYPE=Release -DDEMO_NAME=simple_on_word2vec -DPADDLE_LIB=path_to_the_patddle\paddle_fluid.lib`

Note:

-DDEMO_NAME is the file to be built

-DPADDLE_LIB is the path of fluid_install_dir, for example:
-DPADDLE_LIB=D:\fluid_install_dir


Cmake can be [downloaded at official site](https://cmake.org/download/) and added to environment variables.

After the execution, the directory build is shown in the picture below. Then please open the solution file that which the arrow points at:

<p align="center">
<img src="https://raw.githubusercontent.com/PaddlePaddle/FluidDoc/develop/doc/fluid/advanced_usage/deploy/inference/image/image3.png">
</p>

Modify the attribute of build as `/MT` :

<p align="center">
<img src="https://raw.githubusercontent.com/PaddlePaddle/FluidDoc/develop/doc/fluid/advanced_usage/deploy/inference/image/image4.png">
</p>

<p align="center">
<img src="https://raw.githubusercontent.com/PaddlePaddle/FluidDoc/develop/doc/fluid/advanced_usage/deploy/inference/image/image5.png">
</p>

Modify option of building and generating as `Release` .

<p align="center">
<img src="https://raw.githubusercontent.com/PaddlePaddle/FluidDoc/develop/doc/fluid/advanced_usage/deploy/inference/image/image6.png">
</p>

<p align="center">
<img src="https://raw.githubusercontent.com/PaddlePaddle/FluidDoc/develop/doc/fluid/advanced_usage/deploy/inference/image/image7.png">
</p>

In the dependent packages provided, please copy openblas and model files under Release directory to the directory of Release built and generated.

<p align="center">
<img src="https://raw.githubusercontent.com/PaddlePaddle/FluidDoc/develop/doc/fluid/advanced_usage/deploy/inference/image/image8.png">
</p>

Enter into Release in cmd and run:

  1.  Open GLOG

  	`set GLOG_v=100`

  2.  Start inference

  	`simple_on_word2vec.exe --dirname=.\word2vec.inference.model`

<p align="center">
<img src="https://raw.githubusercontent.com/PaddlePaddle/FluidDoc/develop/doc/fluid/advanced_usage/deploy/inference/image/image9.png">
</p>
