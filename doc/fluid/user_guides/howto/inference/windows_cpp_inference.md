Windows环境模型预测使用说明
===========================

环境部署
--------

### 硬件环境

测试环境硬件配置：

| CPU   |      I7-8700K      |
|----------|:-------------:|
| 内存 |  16G |
| 硬盘 |  1T hdd + 256G ssd |
| 显卡 |  GTX1080 8G |

测试环境操作系统使用win10 Version 18.03 版本。下载地址：

### 环境配置步骤

**一定要严格按照安装步骤顺序，否则会安装失败！**

**安装vs2015**

安装vs2015，安装选项中选择安装内容时勾选自定义，把关于c，c++，vc++的功能都安装上。下载地址：

**安装CUDA8**

需要去NVIDIA官网[https://www.geforce.cn/drivers](https://www.geforce.cn/drivers)
下载显卡对应的驱动。推荐391版本
<p align="center">
 <img src="https://raw.githubusercontent.com/PaddlePaddle/FluidDoc/develop/doc/fluid/user_guides/howto/inference/image/image1.png" >
</p>
安装时需要勾选自定义，勾选安装全部。

验证安装需要进入cmd中，输入nvcc -V查看。
<p align="center">
<img src="https://raw.githubusercontent.com/PaddlePaddle/FluidDoc/develop/doc/fluid/user_guides/howto/inference/image/image2.png">
</p>

如果有显卡安装驱动，也可以选择直接安装CUDA8.0，[https://developer.nvidia.com/cuda-80-ga2-download-archive](https://developer.nvidia.com/cuda-80-ga2-download-archive)

**安装CUDNN**

安装CUDNN只需要将文件中CUDNN
7下的文件复制到对应的CUDA安装目录下。文件名，cudnn-8.0-windows10-x64-v7.zip。这里提供了cudnn
7
64位的版本。需要其他版本可在[https://developer.nvidia.com/cudnn](https://developer.nvidia.com/cudnn)
下载。

预测demo使用
------------

解压Paddle，Release，fluid\_install\_dir压缩包。

进入Paddle/paddle/fluid/inference/api/demo\_ci目录，新建build目录并进入，然后使用cmake生成vs2015的solution文件。
指令为：
```cmake
cmake .. -G \"Visual Studio 14 2015 Win64\" -DWITH\_GPU=ON
-DWITH\_MKL=OFF -DWITH\_STATIC\_LIB=ON -DCMAKE\_BUILD\_TYPE=Release
-DDEMO\_NAME=simple\_on\_word2vec
-DPADDLE\_LIB=D:\\to\_the\_paddle\_fluid.lib
-DCUDA\_LIB=D:\\CUDA\\v8.0\\lib\\x64
```

注：

-DDEMO\_NAME 是要编译的文件

-DPADDLE\_LIB 是fluid\_install\_dir路径，例如
-DPADDLE\_LIB=D:\\fluid\_install\_dir

-DCUDA\_LIB 是CUDA安装目录对应的文件夹

Cmake可以在官网进行下载，并添加到环境变量中。[[https://cmake.org/download/]{.underline}](https://cmake.org/download/)

执行完毕后，build目录如图所示，打开 箭头指向的solution文件：

<p align="center">
<img src="https://raw.githubusercontent.com/PaddlePaddle/FluidDoc/develop/doc/fluid/user_guides/howto/inference/image/image3.png">
</p>

修改编译属性为/MT：

<p align="center">
<img src="https://raw.githubusercontent.com/PaddlePaddle/FluidDoc/develop/doc/fluid/user_guides/howto/inference/image/image4.png">
</p>

<p align="center">
<img src="https://raw.githubusercontent.com/PaddlePaddle/FluidDoc/develop/doc/fluid/user_guides/howto/inference/image/image5.png">
</p>

编译生成选项改成Release。

<p align="center">
<img src="https://raw.githubusercontent.com/PaddlePaddle/FluidDoc/develop/doc/fluid/user_guides/howto/inference/image/image6.png">
</p>

<p align="center">
<img src="https://raw.githubusercontent.com/PaddlePaddle/FluidDoc/develop/doc/fluid/user_guides/howto/inference/image/image7.png">
</p>

将提供的依赖包中，Release下的openblas和模型文件拷贝到编译生成的Release下。

<p align="center">
<img src="https://raw.githubusercontent.com/PaddlePaddle/FluidDoc/develop/doc/fluid/user_guides/howto/inference/image/image8.png">
</p>

通过cmd进到Release目录执行：

  1.  开启GLOG

  	set GLOG\_v=3

  2.  进行预测

  	simple\_on\_word2vec.exe \--dirname=.\\word2vec.inference.model

<p align="center">
<img src="https://raw.githubusercontent.com/PaddlePaddle/FluidDoc/develop/doc/fluid/user_guides/howto/inference/image/image9.png">
</p>

**FAQ：**

路径中尽量不要包含空格，例如发现CUDA\_LIB路径是Program
Files(x86)可能会出错。可以将CUDA拷贝到一个新位置（这里直接拷贝就行）
