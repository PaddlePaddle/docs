# 安装与编译




## 下载速度慢

##### Q: pip install过于缓慢

+ 问题描述:

使用pip或pip3.5/pip3.6/pip3.7 install paddlepaddle时出现下载过慢的情况

+ 问题解答:

可以在使用pip的时候在后面加上-i参数，指定pip源，使用国内源加速：

Python2:

`pip install paddlepaddle -i http://pypi.douban.com/simple/`

Python3:

`pip3 install paddlepaddle -i http://pypi.douban.com/simple/`

这里也可以将 -i 后的参数换成：https://mirrors.aliyun.com/pypi/simple/

##### Q: github下载耗时

+ 问题描述

使用`git clone https://github.com/PaddlePaddle/models.git`命令下载models目录耗时数个小时

+ 问题解答

有两种加速方法（需要管理员权限root/administration）:

1. 编辑hosts 文件

	Linux/Mac：`vim/etc/hosts`

	Windows：双击 `C:\Windows\System32\drivers\etc\hosts`

2. 添加下面两行内容到hosts文件中

	`151.101.72.249 github.global.ssl.fastly.net`

	`192.30.253.112 github.com`


## 环境问题

##### Q: CPU版本可运行，GPU版本运行失败

+ 问题描述

版本为paddlepaddle_gpu-0.14.0.post87-cp27-cp27mu-manylinux1_x86_64.whl，跑一个简单的测试程序，出现Segmentation fault。其中 如果place为cpu，可以正常输出，改成gpu则core。

+ 问题解答

安装版本为`paddlepaddle_gpu-0.14.0.post87-cp27-cp27mu-manylinux1_x86_64.whl`，其中post87是指在CUDA8.0、cudnn7.0编译的，请确定您机器上是否安装了对应版本的cuDNN。造成问题描述中现象的情况通常可能是环境不匹配导致的。

##### Q: 可以用 IDE 吗？

+ 问题解答

当然可以，因为源码就在本机上。IDE 默认调用 make 之类的程序来编译源码，我们只需要配置 IDE 来调用 Docker 命令编译源码即可。

很多 PaddlePaddle 开发者使用 Emacs。他们在自己的 `~/.emacs` 配置文件里加两行:

`global-set-key "\C-cc" 'compile`

`setq compile-command "docker run --rm -it -v $(git rev-parse --show-toplevel):/paddle paddle:dev"`

就可以按 `Ctrl-C` 和 `c` 键来启动编译了。

##### Q: 可以并行编译吗？

+ 问题解答

是的。我们的 Docker image 运行一个 [Bash 脚本](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/scripts/paddle_build.sh)。这个脚本调用`make -j$(nproc)` 来启动和 CPU 核一样多的进程来并行编译。

##### Q: 磁盘不够？

+ 问题解答

在本文中的例子里，`docker run` 命令里都用了 `--rm` 参数，这样保证运行结束之后的 containers 不会保留在磁盘上。可以用 `docker ps -a` 命令看到停止后但是没有删除的 containers。`docker build` 命令有时候会产生一些中间结果，是没有名字的 images，也会占用磁盘。


## CPU

### 源代码编译问题

#### Linux系统

##### Q: CentOS6下如何编译python2.7为共享库

+ 问题解答

使用以下指令：

```
./configure --prefix=/usr/local/python2.7 --enable-shared
make && make install
```

##### Q: 安装swig后找不到swig

+ 问题描述

ubuntu14编译安装时已经安装了swig，之后在虚环境中make编译时报找不到swig错误。

+ 问题解答

安装时没有严格按照官网安装流程的顺序安装，退出虚环境再安装一次swig。

##### Q: 源代码编译安装后出现版本错误

+ 问题描述

在Liunx环境上，通过编译源码的方式安装PaddlePaddle，当安装成功后，运行 `paddle version`, 出现 `PaddlePaddle 0.0.0`？

+ 问题解答

如果运行 `paddle version`, 出现`PaddlePaddle 0.0.0`；或者运行 `cmake ..`，出现

```bash
CMake Warning at cmake/version.cmake:20 (message):
Cannot add paddle version from git tag
```

在dev分支下这个情况是正常的，在release分支下通过export PADDLE_VERSION=对应版本号来解决。

##### Q: Ubuntu编译时大量代码段不能识别

+ 问题解答

这可能是由于cmake版本不匹配造成的，请在gcc的安装目录下使用以下指令：

```
apt install gcc-4.8 g++-4.8
cp gcc gcc.bak
cp g++ g++.bak
rm gcc
rm g++
ln -s gcc-4.8 gcc
ln -s g++-4.8 g++
```

#### MacOS系统

##### Q: 在 Windows/MacOS 上编译很慢？

+ 问题解答

Docker 在 Windows 和 MacOS 都可以运行。不过实际上是运行在一个Linux虚拟机上。可能需要注意给这个虚拟机多分配一些 CPU 和内存，以保证编译高效。具体做法请参考[issue627](https://github.com/PaddlePaddle/Paddle/issues/627)。

##### Q: 编译develop分支代码后出现No such file or directory

+ 问题描述

MacOS本地编译PaddlePaddle github上develop分支的代码出现，出现No such file or directory错误？

+ 问题解答

因为此时develop分支上Generating build/.timestamp这一步涉及的代码还在进行修改，所以并不能保证稳定，建议切换回稳定分支进行编译安装。

可以通过执行如下命令将分支切换到0.14.0进行编译:

```bash
cd Paddle
git checkout -b release/1.1
cd build &&  rm -rf *
cmake .. -DWITH_FLUID_ONLY=ON -DWITH_GPU=OFF -DWITH_TESTING=OFF
make -j4
```

编译成功后的结果如图：

![](https://user-images.githubusercontent.com/17102274/42515418-4fb71e56-848e-11e8-81c6-da2a5553a27a.png)

##### Q: 找不到各种module

+ 问题描述

在MacOSX上从源码编译，最后`cmake ..`时

`Could NOT find PY_google.protobuf (missing: PY_GOOGLE.PROTOBUF)
CMake Error at cmake/FindPythonModule.cmake:27 (message):
python module google.protobuf is not found`

若通过-D设置路径后，又会有其他的如`Could not find PY_wheel`等其他找不到的情况

+ 问题解答

![](https://cloud.githubusercontent.com/assets/728699/19915727/51f7cb68-a0ef-11e6-86cc-febf82a07602.png)

如上，当cmake找到python解释器和python库时，如果安装了许多pythons，它总会找到不同版本的Python。在这种情况下，您应该明确选择应该使用哪个python。
通过cmake显式设置python包。只要确保python libs和python解释器是相同的python可以解决所有这些问题。当这个python包有一些原生扩展时，例如numpy，显式set python包可能会失败。

##### Q: `ld terminated with signal 9 [Killed]`错误

+ 问题描述

在MacOS下，本地直接编译安装PaddlePaddle遇到`collect2: ld terminated with signal 9 [Killed]` ？

+ 问题解答

该问题是由磁盘空间不足造成的，你的硬盘要有30G+的空余空间，请尝试清理出足够的磁盘空间，重新安装。

##### Q: Error 2

+ 问题描述

MacOS本机直接通过源码编译的方式安装PaddlePaddle出现`[paddle/fluid/platform/CMakeFiles/profiler_py_proto.dir/all] Error 2`？

+ 报错截图

![](https://user-images.githubusercontent.com/17102274/42515350-28c055ce-848e-11e8-9b90-c294b375d8a4.png)

+ 问题解答

使用cmake版本为3.4则可。自行编译建议GCC版本:4.8、5.4以及更高。

##### Q: `wget: command not found`

+ 问题描述

MacOS 10.12下编译PaddlePaddle出现`/bin/sh: wget: command not found`，如何解决？

+ 问题解答

报错的原因从报错输出的信息中可以发现，即没有有找到wget命令，安装wget则可，安装命令如下：

```bash
brew install wget
```

##### Q: `Configuring incomplete, errors occured!`

+ 问题描述

以源码方式在MacOS上安装时，出现`Configuring incomplete, errors occured!`？

+ 问题解答

安装PaddlePaddle编译时需要的各种依赖则可，如下：

```bash
pip install wheel
brew install protobuf@3.1
pip install protobuf==3.1.0
```

如果执行pip install protobuf==3.1.0时报错，输出下图内容：

![](https://user-images.githubusercontent.com/17102274/42515286-fb7a7b76-848d-11e8-931a-a7f61bd6374b.png)

从图中可以获得报错的关键为`Cannot uninstall 'six'`，那么解决方法就是先安装好`six`，再尝试安装`protobuf 3.1.0`如下：

```bash
easy_install -U six
pip install protobuf==3.1.0
```

##### Q: 基于Docker容器编译 VS MacOS本机编译

+ 问题描述

PaddlePaddle官方文档中，关于MacOS下安装PaddlePaddle只提及了MacOS中使用Docker环境安装PaddlePaddle的内容，没有Mac本机安装的内容？

+ 问题解答

基于Docker容器编译PaddlePaddle与本机上直接编译PaddlePaddle，所使用的编译执行命令是不一样的，但是官网仅仅给出了基于Docker容器编译PaddlePaddle所执行的命令。

1.基于Docker容器编译PaddlePaddle，需要执行：

```bash
# 1. 获取源码
git clone https://github.com/PaddlePaddle/Paddle.git
cd Paddle
# 2. 可选步骤：源码中构建用于编译PaddlePaddle的Docker镜像
docker build -t paddle:dev .
# 3. 执行下面的命令编译CPU-Only的二进制
docker run -it -v $PWD:/paddle -e "WITH_GPU=OFF" -e "WITH_TESTING=OFF" paddlepaddle/paddle_manylinux_devel:cuda8.0_cudnn5 bash -x /paddle/paddle/scripts/paddle_build.sh build
# 4. 或者也可以使用为上述可选步骤构建的镜像（必须先执行第2步）
docker run -it -v $PWD:/paddle -e "WITH_GPU=OFF" -e "WITH_TESTING=OFF" paddle:dev
```
2.直接在本机上编译PaddlePaddle，需要执行：
```bash
# 1. 使用virtualenvwrapper创建python虚环境并将工作空间切换到虚环境
mkvirtualenv paddle-venv
workon paddle-venv
# 2. 获取源码
git clone https://github.com/PaddlePaddle/Paddle.git
cd Paddle
# 3. 执行下面的命令编译CPU-Only的二进制
mkdir build && cd build
cmake .. -DWITH_GPU=OFF -DWITH_TESTING=OFF
make -j$(nproc)
```

更详细的内容，请参考[官方文档](http://paddlepaddle.org/documentation/docs/zh/1.4/beginners_guide/install/install_MacOS.html)

### 安装冲突

##### Q: Error-平台不支持

+ 问题描述

安装PaddlePaddle过程中，出现`paddlepaddle\*.whl is not a supported wheel on this platform`？

+ 问题解答

`paddlepaddle\*.whl is not a supported wheel on this platform`表示你当前使用的PaddlePaddle不支持你当前使用的系统平台，即没有找到和当前系统匹配的paddlepaddle安装包。最新的paddlepaddle python安装包支持Linux x86_64和MacOS 10.12操作系统，并安装了python 2.7和pip 9.0.1。
请先尝试安装最新的pip，方法如下：

```bash
pip install --upgrade pip
```

如果还不行，可以执行 `python -c "import pip; print(pip.pep425tags.get_supported())"` 获取当前系统支持的python包的后缀，
并对比是否和正在安装的后缀一致。

如果系统支持的是 `linux_x86_64` 而安装包是 `manylinux1_x86_64` ，需要升级pip版本到最新；

如果系统支持 `manylinux1_x86_64` 而安装包（本地）是 `linux_x86_64` ，可以重命名这个whl包为 `manylinux1_x86_64` 再安装。

##### Q: NumPy & Python冲突

+ 问题描述

因为需要安装numpy等包，但在Mac自带的Python上无法安装，权限错误导致难以将PaddlePaddle正常安装到Mac本地？

+ 问题解答

Mac上对自带的Python和包有严格的权限保护，最好不要在自带的Python上安装。建议用virtualenv建立一个新的Python环境来操作。virtualenv的基本原理是将机器上的Python运行所需的运行环境完整地拷贝一份。我们可以在一台机器上制造多份拷贝，并在这多个拷贝之间自由切换，这样就相当于在一台机器上拥有了多个相互隔离、互不干扰的Python环境。

下面使用virtualenv为Paddle生成一个专用的Python环境。

安装virtualenv，virtualenv本身也是Python的一个包，可以用pip进行安装：

```
sudo -H pip install virtualenv
```

由于virtualenv需要安装给系统自带的Python，因此需要使用sudo权限。接着使用安装好的virtualenv创建一个新的Python运行环境：

```
virtualenv --no-site-packages paddle
```

--no-site-packages 参数表示不拷贝已有的任何第三方包，创造一个完全干净的新Python环境。后面的paddle是我们为这个新创建的环境取的名字。执行完这一步后，当前目录下应该会出现一个名为paddle（或者你取的其他名字）的目录。这个目录里保存了运行一个Python环境所需要的各种文件。
启动运行环境：

```
source paddle/bin/activate
```

执行后会发现命令提示符前面增加了(paddle)字样，说明已经成功启动了名为‘paddle’的Python环境。执行which python，可以发现使用的已经是刚刚创建的paddle目录下的Python。在这个环境中，我们可以自由地进行PaddlePaddle的安装、使用和开发工作，无需担心对系统自带Python的影响。
如果我们经常使用Paddle这个环境，我们每次打开终端后都需要执行一下source paddle/bin/activate来启动环境，比较繁琐。为了简便，可以修改终端的配置文件，来让终端每次启动后自动启动特定的Python环境。
执行:

```
vi ~/.bash_profile
```

打开终端配置文件，并在文件的最后添加一行：

```
source paddle/bin/activate
```

这样，每次打开终端时就会自动启动名为‘paddle’的Python环境了。

### 安装后无法使用

##### Q: 安装后无法import paddle.fluid

+ 问题描述

MacOS下安装PaddlePaddle后import paddle.fluid出现`Fatal Python error: PyThreadState_Get: no current thread running`错误

+ 问题解答

For Python2.7.x （install by brew): 请使用`export LD_LIBRARY_PATH=/usr/local/Cellar/python@2/2.7.15_1/Frameworks/Python.framework/Versions/2.7 && export DYLD_LIBRARY_PATH=/usr/local/Cellar/python@2/2.7.15_1/Frameworks/Python.framework/Versions/2.7`

For Python2.7.x （install by Python.org): 请使用`export LD_LIBRARY_PATH=/Library/Frameworks/Python.framework/Versions/2.7 && export DYLD_LIBRARY_PATH=/Library/Frameworks/Python.framework/Versions/2.7`

For Python3.5.x （install by Python.org): 请使用`export LD_LIBRARY_PATH=/Library/Frameworks/Python.framework/Versions/3.5/ && export DYLD_LIBRARY_PATH=/Library/Frameworks/Python.framework/Versions/3.5/`

##### Q: CPU版本训练中自动退出

+ 问题描述

成功安装了PaddlePaddle CPU版本后，使用Paddle训练模型，训练过程中，Paddle会自动退出，gdb显示Illegal instruction？

+ 问题解答

CPU版本PaddlePaddle自动退出的原因通常是因为所在机器不支持AVX2指令集而主动abort。简单的判断方法：
用gdb-7.9以上版本（因编译C++文件用的工具集是gcc-4.8.2，目前只知道gdb-7.9这个版本可以debug gcc4编译出来的目标文件）：

```bash
$ /path/to/gdb -iex "set auto-load safe-path /" -iex "set solib-search-path /path/to/gcc-4/lib" /path/to/python -c core.xxx
```

在gdb界面：

```bash
(gdb) disas
```

找到箭头所指的指令，例如：

```bash
   0x00007f381ae4b90d <+3101>:  test   %r8,%r8
=> 0x00007f381ae4b912 <+3106>:  vbroadcastss %xmm0,%ymm1
   0x00007f381ae4b917 <+3111>:  lea    (%r12,%rdx,4),%rdi
```

然后google一下这个指令需要的指令集。上面例子中的带xmm和ymm操作数的vbroadcastss指令只在AVX2中支持
然后看下自己的CPU是否支持该指令集

```bash
cat /proc/cpuinfo | grep flags | uniq | grep avx --color
```

如果没有AVX2，就表示确实是指令集不支持引起的主动abort。

如果没有AVX2指令集，就需要要安装不支持AVX2指令集版本的PaddlePaddle，默认安装的PaddlePaddle是支持AVX2指令集的，因为AVX2可以加速模型训练的过程，更多细节可以参考[安装文档](http://paddlepaddle.org/documentation/docs/zh/1.4/beginners_guide/install/index_cn.html)。

##### Q: Python相关单元测试失败

+ 问题描述

安装完了PaddlePaddle后，出现以下python相关的单元测试都过不了的情况：

```
24 - test_PyDataProvider (Failed)
26 - test_RecurrentGradientMachine (Failed)
27 - test_NetworkCompare (Failed)
28 - test_PyDataProvider2 (Failed)
32 - test_Prediction (Failed)
33 - test_Compare (Failed)
34 - test_Trainer (Failed)
35 - test_TrainerOnePass (Failed)
36 - test_CompareTwoNets (Failed)
37 - test_CompareTwoOpts (Failed)
38 - test_CompareSparse (Failed)
39 - test_recurrent_machine_generation (Failed)
40 - test_PyDataProviderWrapper (Failed)
41 - test_config_parser (Failed)
42 - test_swig_api (Failed)
43 - layers_test (Failed)
```

并且查询PaddlePaddle单元测试的日志，提示：

```
paddle package is already in your PYTHONPATH. But unittest need a clean environment.
Please uninstall paddle package before start unittest. Try to 'pip uninstall paddle'.
```

+ 问题解答

卸载PaddlePaddle包 `pip uninstall paddle`, 清理掉老旧的PaddlePaddle安装包，使得单元测试有一个干净的环境。如果PaddlePaddle包已经在python的site-packages里面，单元测试会引用site-packages里面的python包，而不是源码目录里 `/python` 目录下的python包。同时，即便设置 `PYTHONPATH` 到 `/python` 也没用，因为python的搜索路径是优先已经安装的python包。



## GPU

### 安装过程中报错

##### Q: Error: Can not load core_noavx.* .ImportError

+ 问题描述

为了更好的支持paddle安装包在不同指令集（AVX和没有AVX）的机器上运行，近期对paddle核心core文件进行了修改（有重命名），可能会导致开发者遇到类似如下warning或者错误：

```
WARNING: Can not import avx core. You may not build with AVX, but AVX is supported on local machine, you could build paddle WITH_AVX=ON to get bettwe performance. Error: Can not load core_noavx.* .ImportError
```
+ 问题解读

这条信息分为2个，前者是warning，是追求性能的warning，普通情况可以忽略。后者是，Error：往后才是导致不能继续运行的地方。

+ 问题解答

1. 更新最新代码，如上图的错误代表您还不是最新代码，请更新之后尝试。

2. 如果您机器上之前本来就安装有paddlepaddle，请使用pip install -U paddlepaddle， 加上-U选项明确代表升级。

3. 如果问题还存在，可能问题原因是之前build缓存没有删除导致，可以make clean，删除build目录下的python目录从而删除原有缓存，重新编译安装即可。

4. 如果仍然有问题，当然，通常到这里就已经不是paddle本身的问题了，并且该错误跟AVX本身没有任何关系，请怀疑下您的运行环境是否和编译环境一致，包括glibc，python版本等信息。或者，是您的加了什么代码，没有正确加到pybind导致的错误。
    1. 请仔细查看错误信息，会提示缺失或者load错误的原因。
    2. 请确认安装后的python目录下（通常会在/usr/local/lib/python2.7/dist-packages/paddle/fluid）中是否有core_avx.so或者core_noavx.so，这两文件其中一个，或者两个都有。如果都没有，或者只有core.so那说明第一和二步没有正确执行。并请仔细查看python输出的load失败的报错信息。通常这个是因为编译和运行不在一个环境导致的错误，比如glibc版本不匹配等，这个与本次升级无关。

##### Q: 报错“nccl.h找不到”

+ 问题解答：

请[安装nccl2](https://developer.nvidia.com/nccl/nccl-download)

#### 安装过程中cuDNN报错

##### Q: CUDNN_STATUS_NOT_INITIALIZED

+ 问题描述

遇到如下cuDNN报错如何解决？

```
CUDNN_STATUS_NOT_INITIALIZED at [/paddle/paddle/fluid/platform/device_context.cc:216]
```

+ 问题解答

cuDNN与CUDA版本不一致导致。PIP安装的GPU版本默认使用CUDA 9.0和cuDNN 7编译，请根据您的环境配置选择在官网首页选择对应的安装包进行安装，例如paddlepaddle-gpu==1.2.0.post87 代表使用CUDA 8.0和cuDNN 7编译的1.2.0版本。

#### 安装过程中CUDA报错

##### Q: cuda9.0对应paddle版本

+ 问题描述
cuda9.0需要安装哪一个版本的paddle，安装包在哪?

+ 问题解答

`pip install paddlepaddle-gpu`命令将安装支持CUDA 9.0 cuDNN v7的PaddlePaddle，可以参考[安装说明文档](http://paddlepaddle.org/documentation/docs/zh/1.4/beginners_guide/install/index_cn.html)

##### Q: driver version is insufficient for runtime version

+ 问题描述

在使用PaddlePaddle GPU的Docker镜像的时候，出现 `Cuda Error: CUDA driver version is insufficient for CUDA runtime version`？

+ 问题解答

通常出现 `Cuda Error: CUDA driver version is insufficient for CUDA runtime version`, 原因在于没有把机器上CUDA相关的驱动和库映射到容器内部。

使用nvidia-docker, 命令只需要将docker换为nvidia-docker即可。

更多请参考[nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

#### Docker

使用Docker出现编译错误，请额外参照GitHub上[Issue12079](https://github.com/PaddlePaddle/Paddle/issues/12079)

##### Q: 无法下载golang

+ 问题描述

根据官方文档中提供的步骤安装Docker，无法下载需要的golang，导致`tar: Error is not recoverable: exiting now`？

+ 报错截图

![](https://user-images.githubusercontent.com/17102274/42516245-314346be-8490-11e8-85cc-eb95e9f0e02c.png)

+ 问题解答

由上图可知，生成docker镜像时需要下载[golang](https://storage.googleapis.com/golang/go1.8.1.linux-amd64.tar.gz)，使用者需要保证电脑可以科学上网。

选择下载并使用docker.paddlepaddlehub.com/paddle:latest-devdocker镜像，执行命令如下：

```
git clone https://github.com/PaddlePaddle/Paddle.git
cd Paddle
git checkout -b 0.14.0 origin/release/0.14.0
sudo docker run --name paddle-test -v $PWD:/paddle --network=host -it docker.paddlepaddlehub.com/paddle:latest-dev /bin/bash
```

进入docker编译GPU版本的PaddlePaddle，执行命令如下：

```
mkdir build && cd build
# 编译GPU版本的PaddlePaddle
cmake .. -DWITH_FLUID_ONLY=ON -DWITH_GPU=ON -DWITH_TESTING=ON
make -j$(nproc)
```

通过上面的方式操作后：

![](https://user-images.githubusercontent.com/17102274/42516287-46ccae8a-8490-11e8-9186-985efff3629c.png)

接着安装PaddlePaddle并运行线性回归test_fit_a_line.py程序测试一下PaddlePaddle是安装成功则可

```bash
pip install build/python/dist/*.whl
python python/paddle/fluid/tests/book/test_fit_a_line.py
```

##### Q: 在Docker镜像上，GPU版本的PaddlePaddle运行结果报错

+ 问题描述

![](https://user-images.githubusercontent.com/17102274/42516300-50f04f8e-8490-11e8-95f1-613d3d3f6ca6.png)

![](https://user-images.githubusercontent.com/17102274/42516303-5594bd22-8490-11e8-8c01-55741484f126.png)

+ 问题解答

使用`sudo docker run --name paddle-test -v $PWD:/paddle --network=host -it docker.paddlepaddlehub.com/paddle:latest-dev /bin/bash`命令创建的docker容器仅能支持运行CPU版本的PaddlePaddle。
使用如下命令重新开启支持GPU运行的docker容器：

```
export CUDA_SO="$(\ls /usr/lib64/libcuda* | xargs -I{} echo '-v {}:{}') $(\ls /usr/lib64/libnvidia* | xargs -I{} echo '-v {}:{}')"
export DEVICES=$(\ls /dev/nvidia* | xargs -I{} echo '--device {}:{}')
sudo docker run ${CUDA_SO} ${DEVICES} --rm --name paddle-test-gpu -v /usr/bin/nvidia-smi:/usr/bin/nvidia-smi -v $PWD:/paddle --network=host -it docker.paddlepaddlehub.com/paddle:latest-dev /bin/bash
```

进入docker之后执行如下命令进行PaddlePaddle的安装及测试运行：

```
export LD_LIBRARY_PATH=/usr/lib64:/usr/local/lib:$LD_LIBRARY_PATH
pip install build/python/dist/*.whl
python python/paddle/fluid/tests/book/test_fit_a_line.py
```

### 安装后无法使用

##### Q: GPU安装成功，无法import

+ 问题描述

使用 `sudo nvidia-docker run --name Paddle -it -v $PWD:/work hub.baidubce.com/paddlepaddle/paddle:latest-gpu-cuda8.0-cudnn7 /bin/bash`，安装成功后，出现如下问题

```
import paddle.fluid
*** Aborted at 1539682149 (unix time) try "date -d @1539682149" if you are using GNU date ***
PC: @ 0x0 (unknown)
*** SIGILL (@0x7f6ac6ea9436) received by PID 16 (TID 0x7f6b07bc7700) from PID 18446744072751846454; stack trace: ***
```

+ 问题解答

请先确定一下机器是否支持AVX2指令集，如果不支持，请按照相应的不支持AVX2指令集的PaddlePaddle，可以解决该问题。

##### Q: 曾经运行成功，现在运行失败

+ 问题描述

使用 `pip install paddlepaddle-gpu==0.14.0.post87`命令在公司内部开发GPU机器上安装PaddlePaddle，按照官网安装：pip install paddlepaddle-gpu==0.14.0.post87，执行 import paddle.fluid as fluid 失败。奇怪的是，同样的环境下，上周运行成功，这周确运行失败，求解答？

+ 问题解答

这通常是GPU显存不足导致的，请检查一下机器的显存，确保显存足够后再尝试import paddle.fluid

##### Q: 安装成功后，示例运行失败
+ 问题描述

安装的是cuda9.0和cudnn7.0，默认安装的是0.14.0.post87，训练一个手写数据那个例子的时候报错？

+ 问题解答

该问题通常是GPU显存不足造成的，请在显存充足的GPU服务器上再次尝试则可。可以检查一下机器的显存使用情况。

方法如下：

```bash
test@test:~$ nvidia-smi
Tue Jul 24 08:24:22 2018       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 384.130                Driver Version: 384.130                   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX 960     Off  | 00000000:01:00.0  On |                  N/A |
| 22%   52C    P2   100W / 120W |   1757MiB /  1994MiB |     98%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    0      1071      G   /usr/lib/xorg/Xorg                           314MiB |
|    0      1622      G   compiz                                       149MiB |
|    0      2201      G   fcitx-qimpanel                                 7MiB |
|    0     15304      G   ...-token=58D78B2D4A63DAE7ED838021B2136723    74MiB |
|    0     15598      C   python                                      1197MiB |
+-----------------------------------------------------------------------------+
```

## 卸载问题

##### Q: 报错`Cannot uninstall 'six'.`

+ 问题描述

此问题可能与系统中已有Python有关，请使用`pip install paddlepaddle --ignore-installed six`（CPU）或`pip install paddlepaddle --ignore-installed six`（GPU）解决
