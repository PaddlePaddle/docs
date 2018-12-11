Windows环境模型预测
===========================

环境部署
--------

### 硬件环境

测试环境硬件配置：

| CPU      |      I7-8700K      |
|:---------|:-------------------|
| 内存 | 16G               |
| 硬盘 | 1T hdd + 256G ssd |
| 显卡 | GTX1080 8G        |

测试环境操作系统使用 win10 家庭版本。

### 环境配置步骤

**请您严格按照以下步骤进行安装，否则可能会导致安装失败！**

**安装Visual Studio 2015 update3**

安装Visual Studio 2015，安装选项中选择安装内容时勾选自定义，选择安装全部关于c，c++，vc++的功能。


预测demo使用
------------

解压Paddle，Release，fluid_install_dir压缩包。

进入Paddle/paddle/fluid/inference/api/demo_ci目录，新建build目录并进入，然后使用cmake生成vs2015的solution文件。
指令为：

`cmake .. -G "Visual Studio 14 2015 Win64" -DWITH_GPU=OFF -DWITH_MKL=OFF -DWITH_STATIC_LIB=ON -DCMAKE_BUILD_TYPE=Release -DDEMO_NAME=simple_on_word2vec -DPADDLE_LIB=path_to_the_patddle\paddle_fluid.lib`

注：

-DDEMO_NAME 是要编译的文件

-DPADDLE_LIB 是fluid_install_dir路径，例如
-DPADDLE_LIB=D:\fluid_install_dir


Cmake可以在[官网进行下载](https://cmake.org/download/)，并添加到环境变量中。

执行完毕后，build 目录如图所示，打开箭头指向的 solution 文件：

<p align="center">
<img src="https://raw.githubusercontent.com/PaddlePaddle/FluidDoc/develop/doc/fluid/advanced_usage/deploy/inference/image/image3.png">
</p>

修改编译属性为 `/MT` ：

<p align="center">
<img src="https://raw.githubusercontent.com/PaddlePaddle/FluidDoc/develop/doc/fluid/advanced_usage/deploy/inference/image/image4.png">
</p>

<p align="center">
<img src="https://raw.githubusercontent.com/PaddlePaddle/FluidDoc/develop/doc/fluid/advanced_usage/deploy/inference/image/image5.png">
</p>

编译生成选项改成 `Release` 。

<p align="center">
<img src="https://raw.githubusercontent.com/PaddlePaddle/FluidDoc/develop/doc/fluid/advanced_usage/deploy/inference/image/image6.png">
</p>

<p align="center">
<img src="https://raw.githubusercontent.com/PaddlePaddle/FluidDoc/develop/doc/fluid/advanced_usage/deploy/inference/image/image7.png">
</p>

将提供的依赖包中，Release下的openblas和模型文件拷贝到编译生成的Release下。

<p align="center">
<img src="https://raw.githubusercontent.com/PaddlePaddle/FluidDoc/develop/doc/fluid/advanced_usage/deploy/inference/image/image8.png">
</p>

通过cmd进到Release目录执行：

  1.  开启GLOG

  	`set GLOG_v=100`

  2.  进行预测

  	`simple_on_word2vec.exe --dirname=.\word2vec.inference.model`

<p align="center">
<img src="https://raw.githubusercontent.com/PaddlePaddle/FluidDoc/develop/doc/fluid/advanced_usage/deploy/inference/image/image9.png">
</p>

