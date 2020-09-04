源码编译
========

什么时候需要源码编译？
--------------

深度学习的发展十分迅速，对科研或工程人员来说，可能会遇到一些需要自己开发op的场景，可以在python层面编写op，但如果对性能有严格要求的话则必须在C++层面开发op，对于这种情况，需要用户源码编译飞桨，使之生效。
此外对于绝大多数使用C++将模型部署上线的工程人员来说，您可以直接通过飞桨官网下载已编译好的预测库，快捷开启飞桨使用之旅。`飞桨官网 <https://www.paddlepaddle.org.cn/documentation/docs/zh/advanced_guide/inference_deployment/inference/build_and_install_lib_cn.html>`_ 提供了多个不同环境下编译好的预测库。如果用户环境与官网提供环境不一致（如cuda 、cudnn、tensorrt版本不一致等），或对飞桨源代码有修改需求，或希望进行定制化构建，可查阅本文档自行源码编译得到预测库。

编译原理
---------

**一：目标产物**

飞桨框架的源码编译包括源代码的编译和链接，最终生成的目标产物包括：

 - 含有 C++ 接口的头文件及其二进制库：用于C++环境，将文件放到指定路径即可开启飞桨使用之旅。
 - Python Wheel 形式的安装包：用于Python环境，此安装包需要参考 `飞桨安装教程 <https://www.paddlepaddle.org.cn/>`_ 进行安装操作。也就是说，前面讲的pip安装属于在线安装，这里属于本地安装。

**二：基础概念**

飞桨主要由C++语言编写，通过pybind工具提供了Python端的接口，飞桨的源码编译主要包括编译和链接两步。
* 编译过程由编译器完成，编译器以编译单元（后缀名为 .cc 或 .cpp 的文本文件）为单位，将 C++ 语言 ASCII 源代码翻译为二进制形式的目标文件。一个工程通常由若干源码文件组织得到，所以编译完成后，将生成一组目标文件。
* 链接过程使分离编译成为可能，由链接器完成。链接器按一定规则将分离的目标文件组合成一个能映射到内存的二进制程序文件，并解析引用。由于这个二进制文件通常包含源码中指定可被外部用户复用的函数接口，所以也被称作函数库。根据链接规则不同，链接可分为静态和动态链接。静态链接对目标文件进行归档；动态链接使用地址无关技术，将链接放到程序加载时进行。
配合包含声明体的头文件（后缀名为 .h 或 .hpp），用户可以复用程序库中的代码开发应用。静态链接构建的应用程序可独立运行，而动态链接程序在加载运行时需到指定路径下搜寻其依赖的二进制库。

**三：编译方式**

飞桨框架的设计原则之一是满足不同平台的可用性。然而，不同操作系统惯用的编译和链接器是不一样的，使用它们的命令也不一致。比如，Linux 一般使用 GNU 编译器套件（GCC），Windows 则使用 Microsoft Visual C++（MSVC）。为了统一编译脚本，飞桨使用了支持跨平台构建的 CMake，它可以输出上述编译器所需的各种 Makefile 或者 Project 文件。    
为方便编译，框架对常用的CMake命令进行了封装，如仿照 Bazel工具封装了 cc_binary 和 cc_library ，分别用于可执行文件和库文件的产出等，对CMake感兴趣的同学可在 cmake/generic.cmake 中查看具体的实现逻辑。Paddle的CMake中集成了生成python wheel包的逻辑，对如何生成wheel包感兴趣的同学可参考 `相关文档 <https://packaging.python.org/tutorials/packaging-projects/>`_ 。


编译步骤
-----------

飞桨分为 CPU 版本和 GPU 版本。如果您的计算机没有 Nvidia GPU，请选择 CPU 版本构建安装。如果您的计算机含有 Nvidia GPU（ 1.0 且预装有 CUDA / CuDNN，也可选择 GPU 版本构建安装。本节简述飞桨在常用环境下的源码编译方式，欢迎访问飞桨官网获取更详细内容。请阅读本节内容。

**推荐配置及依赖项**

1、稳定的互联网连接，主频 1 GHz 以上的多核处理器，9 GB 以上磁盘空间。  
2、Python 版本 2.7 或 3.5 以上，pip 版本 9.0 及以上；CMake v3.5 及以上；Git 版本 2.17 及以上。请将可执行文件放入系统环境变量中以方便运行。  
3、GPU 版本额外需要 Nvidia CUDA 9 / 10，CuDNN v7 及以上版本。根据需要还可能依赖 NCCL 和 TensorRT。  


基于Ubuntu 18.04
------------

**一：环境准备**

除了本节开头提到的依赖，在 Ubuntu 上进行飞桨的源码编译，您还需要准备 GCC8 编译器等工具，可使用下列命令安装：

.. code:: shell

	sudo apt-get install gcc g++ make cmake git vim unrar python3 python3-dev python3-pip swig wget patchelf libopencv-dev
	pip3 install numpy protobuf wheel setuptools

若需启用 cuda 加速，需准备 cuda、cudnn、nccl。上述工具的安装请参考 nvidia 官网，以 cuda10.1，cudnn7.6 为例配置 cuda 环境。

.. code:: shell

	# cuda
	sh cuda_10.1.168_418.67_linux.run
	export PATH=/usr/local/cuda-10.1/bin${PATH:+:${PATH}}
	export LD_LIBRARY_PATH=/usr/local/cuda-10.1/${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

	# cudnn
	tar -xzvf cudnn-10.1-linux-x64-v7.6.4.38.tgz
	sudo cp -a cuda/include/cudnn.h /usr/local/cuda/include/
	sudo cp -a cuda/lib64/libcudnn* /usr/local/cuda/lib64/

	# nccl
	# install nccl local deb 参考https://docs.nvidia.com/deeplearning/sdk/nccl-install-guide/index.html
	sudo dpkg -i nccl-repo-ubuntu1804-2.5.6-ga-cuda10.1_1-1_amd64.deb
	# 根据安装提示，还需要执行sudo apt-key add /var/nccl-repo-2.5.6-ga-cuda10.1/7fa2af80.pub
	sudo apt update
	sudo apt install libnccl2 libnccl-dev

	sudo ldconfig


**编译飞桨过程中可能会打开很多文件，Ubuntu 18.04 默认设置最多同时打开的文件数是1024（参见 ulimit -a），需要更改这个设定值。** 


在 /etc/security/limits.conf 文件中添加两行。

.. code:: shell
 
	* hard noopen 102400
	* soft noopen 102400

重启计算机，重启后执行以下指令，请将${user}切换成当前用户名。

.. code:: shell

	su ${user}
	ulimit -n 102400


**二：编译命令**

使用 Git 将飞桨代码克隆到本地，并进入目录，切换到稳定版本（git tag显示的标签名，如v1.7.1）。  
**飞桨使用 develop 分支进行最新特性的开发，使用 release 分支发布稳定版本。在 GitHub 的 Releases 选项卡中，可以看到飞桨版本的发布记录。**  

.. code:: shell

	git clone https://github.com/PaddlePaddle/Paddle.git
	cd Paddle
	git checkout v1.7.1    

下面以 GPU 版本为例说明编译命令。其他环境可以参考“CMake编译选项表”修改对应的cmake选项。比如，若编译 CPU 版本，请将 WITH_GPU 设置为 OFF。


.. code:: shell

	# 创建并进入 build 目录
	mkdir build_cuda && cd build_cuda
	# 执行cmake指令
	cmake -DPY_VERSION=3 \
		-DWITH_TESTING=OFF \
		-DWITH_MKL=ON \
		-DWITH_GPU=ON \
		-DON_INFER=ON \
		-DCMAKE_BUILD_TYPE=RelWithDebInfo \
		..
		
**使用make编译**

make -j4

**编译成功后可在dist目录找到生成的.whl包**

pip3 install python/dist/paddlepaddle-1.7.1-cp36-cp36m-linux_x86_64.whl

**预测库编译**

make inference_lib_dist -j4


**cmake编译环境表**

以下介绍的编译方法都是通用步骤，根据环境对应修改cmake选项即可。

================  ============================================================================  =============================================================
      选项                                            说明                                                                 默认值
================  ============================================================================  =============================================================
WITH_GPU          是否支持GPU                                                                   ON
WITH_AVX          是否编译含有AVX指令集的飞桨二进制文件                                         ON
WITH_PYTHON       是否内嵌PYTHON解释器并编译Wheel安装包                                         ON
WITH_TESTING      是否开启单元测试                                                              OFF
WITH_MKL          是否使用MKL数学库，如果为否，将使用OpenBLAS                                   ON
WITH_SYSTEM_BLAS  是否使用系统自带的BLAS                                                        OFF
WITH_DISTRIBUTE   是否编译带有分布式的版本                                                      OFF
WITH_BRPC_RDMA    是否使用BRPC,RDMA作为RPC协议                                                  OFF
ON_INFER          是否打开预测优化                                                              OFF
CUDA_ARCH_NAME    是否只针对当前CUDA架构编译                                                    All:编译所有可支持的CUDA架构；Auto:自动识别当前环境的架构编译
TENSORRT_ROOT     TensorRT_lib的路径，该路径指定后会编译TRT子图功能eg:/paddle/nvidia/TensorRT/  /usr
================  ============================================================================  =============================================================

基于Windows 10 
-------------------

**一：环境准备**

除了本节开头提到的依赖，在 Windows 10 上编译飞桨，您还需要准备 Visual Studio 2015 Update3 以上版本。本节以 Visual Studio 企业版 2019（C++ 桌面开发，含 MSVC 14.24）、Python 3.8 为例介绍编译过程。

在命令提示符输入下列命令，安装必需的 Python 组件。

.. code:: shell

	pip3 install numpy protobuf wheel` 

**二：编译命令**
 
使用 Git 将飞桨代码克隆到本地，并进入目录，切换到稳定版本（git tag显示的标签名，如v1.7.1）。  
**飞桨使用 develop 分支进行最新特性的开发，使用 release 分支发布稳定版本。在 GitHub 的 Releases 选项卡中，可以看到 Paddle 版本的发布记录。**

.. code:: shell

	git clone https://github.com/PaddlePaddle/Paddle.git
	cd Paddle
	git checkout v1.7.1
	
创建一个构建目录，并在其中执行 CMake，生成解决方案文件 Solution File，以编译 CPU 版本为例说明编译命令，其他环境可以参考“CMake编译选项表”修改对应的cmake选项。

.. code:: shell

	mkdir build
	cd build
	cmake .. -G "Visual Studio 16 2019" -A x64 -DWITH_GPU=OFF -DWITH_TESTING=OFF 
		-DCMAKE_BUILD_TYPE=Release -DPY_VERSION=3
	
.. image:: https://agroup-bos.cdn.bcebos.com/1b21aff9424cb33a98f2d1e018d8301614caedda

使用 Visual Studio 打开解决方案文件，在窗口顶端的构建配置菜单中选择 Release x64，单击生成解决方案，等待构建完毕即可。  

**cmake编译环境表**

================  ============================================================================  =============================================================
      选项                                            说明                                                                 默认值
================  ============================================================================  =============================================================
WITH_GPU          是否支持GPU                                                                   ON
WITH_AVX          是否编译含有AVX指令集的飞桨二进制文件                                         ON
WITH_PYTHON       是否内嵌PYTHON解释器并编译Wheel安装包                                         ON
WITH_TESTING      是否开启单元测试                                                              OFF
WITH_MKL          是否使用MKL数学库，如果为否，将使用OpenBLAS                                   ON
WITH_SYSTEM_BLAS  是否使用系统自带的BLAS                                                        OFF
WITH_DISTRIBUTE   是否编译带有分布式的版本                                                      OFF
WITH_BRPC_RDMA    是否使用BRPC,RDMA作为RPC协议                                                  OFF
ON_INFER          是否打开预测优化                                                              OFF
CUDA_ARCH_NAME    是否只针对当前CUDA架构编译                                                    All:编译所有可支持的CUDA架构；Auto:自动识别当前环境的架构编译
TENSORRT_ROOT     TensorRT_lib的路径，该路径指定后会编译TRT子图功能eg:/paddle/nvidia/TensorRT/  /usr
================  ============================================================================  =============================================================

**结果验证**

**一：python whl包**

编译完毕后，会在 python/dist 目录下生成一个文件名类似 paddlepaddle-1.7.1-cp36-cp36m-linux_x86_64.whl 的 Python Wheel 安装包，安装测试的命令为：  

.. code:: shell

	pip3 install python/dist/paddlepaddle-1.7.1-cp36-cp36m-linux_x86_64.whl

安装完成后，可以使用 python3 进入python解释器，输入以下指令，出现 `Your Paddle Fluid is installed succesfully! ` ，说明安装成功。

.. code:: python

	import paddle.fluid as fluid
	fluid.install_check.run_check()


**二：c++ lib**

预测库编译后，所有产出均位于build目录下的fluid_inference_install_dir目录内，目录结构如下。version.txt 中记录了该预测库的版本信息，包括Git Commit ID、使用OpenBlas或MKL数学库、CUDA/CUDNN版本号。

.. code:: shell

	build/fluid_inference_install_dir
	├── CMakeCache.txt
	├── paddle
	│   ├── include
	│   │   ├── paddle_anakin_config.h
	│   │   ├── paddle_analysis_config.h
	│   │   ├── paddle_api.h
	│   │   ├── paddle_inference_api.h
	│   │   ├── paddle_mkldnn_quantizer_config.h
	│   │   └── paddle_pass_builder.h
	│   └── lib
	│       ├── libpaddle_fluid.a (Linux)
	│       ├── libpaddle_fluid.so (Linux)
	│       └── libpaddle_fluid.lib (Windows)
	├── third_party
	│   ├── boost
	│   │   └── boost
	│   ├── eigen3
	│   │   ├── Eigen
	│   │   └── unsupported
	│   └── install
	│       ├── gflags
	│       ├── glog
	│       ├── mkldnn
	│       ├── mklml
	│       ├── protobuf
	│       ├── xxhash
	│       └── zlib
	└── version.txt


Include目录下包括了使用飞桨预测库需要的头文件，lib目录下包括了生成的静态库和动态库，third_party目录下包括了预测库依赖的其它库文件。

您可以编写应用代码，与预测库联合编译并测试结果。请参 `C++ 预测库 API 使用 <https://aistudio.baidu.com/bjcpu/user/166411/248511/notebooks/248511.ipynb?redirects=1#C++%E9%A2%84%E6%B5%8BAPI>`_ 一节。
