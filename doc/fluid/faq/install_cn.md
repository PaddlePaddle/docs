# 安装类FAQ
*在使用PaddlePaddle遇到问题时，请先查阅文中FAQ；如果下面的FAQ无法解决您的问题，请ISSUE提问，谢谢*

**问题**：Windows环境下，使用pip install时速度慢，如何解决？

**解决方案：**

在pip后面加上参数`-i`指定pip源，使用国内源获取安装包。

**操作步骤：**

1. Python2情况下，使用如下命令安装PaddlePaddle。

   `pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple/`

2. Python3情况下，使用如下命令安装PaddlePaddle。

   `pip3 install paddlepaddle -i https://mirror.baidu.com/pypi/simple/`

您还可以通过如下三个地址获取pip安装包，只需修改 `-i` 后网址即可：

https://pypi.tuna.tsinghua.edu.cn/simple
https://mirrors.aliyun.com/pypi/simple/
https://pypi.douban.com/simple/

------

**问题**：MacOS环境下，使用pip install时报错，`Error：No Matching distribution found for paddlepaddle` ，如何解决？

**问题描述：**

![图片](https://agroup-bos-bj.cdn.bcebos.com/bj-6ddccd0cad4f70363da2be2977508eca64203c8b)

**报错分析**：

用户设置了从阿里云下载 `https://mirrors.aliyun.com/pypi/simple/`，但是 MacOS 对于该地址进行了限制，导致报错。

**解决方法：**

建议通过如下两个地址获取pip安装包，修改 `-i` 后网址即可：

https://pypi.tuna.tsinghua.edu.cn/simple
https://pypi.douban.com/simple/

------

**问题**：使用pip install时报错，`PermissionError: [WinError 5]`，`C:\\program fiels\\python35\\Lib\\site-packages\\graphviz`。 如何解决？


**报错分析：**

用户权限问题导致，由于用户的Python安装到系统文件内（如”Program Files/“），任何的操作都需要管理员权限。

**解决方法：**

选择“以管理员身份运行”运行CMD，重新执行安装过程。

------

**问题：**使用pip install时报错，`ERROR: No matching distribution found for paddlepaddle` ，如何解决？

**问题描述：**

![图片](https://agroup-bos-bj.cdn.bcebos.com/bj-febb18fb78004dc17f18d60a009dc6a8bd907251)

**报错分析：**

Python版本不匹配导致。用户使用的是32位Python，但是对应的32位pip没有PaddlePaddle源。

**解决方法：**

请用户使用64位的Python进行PaddlePaddle安装。

------

**问题：**在GPU版本为`paddlepaddle_gpu-1.8.4.post87-cp27-cp27mu-manylinux1_x86_64.whl`的环境上执行一个程序，出现`Error：Segmentation fault`。如果将`place`修改“cpu”，则程序可正常运行。是什么原因？

**报错分析：**

造成该报错的原因通常是环境不匹配导致的。安装时，GPU版本为`paddlepaddle_gpu-1.8.4.post87-cp27-cp27mu-manylinux1_x86_64.whl`，`post87`表示需要在CUDA8.0、cuDNN7.0进行编译。如果机器上没有安装对应版本的CUDA和cuDNN，会导致执行程序时报错。

此外值得注意的是，配置PaddlePaddle的GPU版本，不仅需要CUDA和cuDNN版本匹配，还需要与PaddlePaddle版本匹配。出现类似错误时请检查这三个程序的版本是否匹配。

**解决方法：**

CUDA的安装可参考：https://docs.nvidia.com/cuda/archive/10.0/index.html；cuDNN的安装可参考：https://docs.nvidia.com/deeplearning/cudnn/install-guide/#install-windows。

------

**问题：**本地使用import paddle时报错，`ModuleNotFoundError:No module named ‘paddle’`，如何解决？

**报错分析：**

原因在于用户的计算机上可能安装了多个版本的Python，而安装PaddlePaddle时的Python和import paddle时的Python版本不一致导致报错。如果用户熟悉PyCharm等常见的IDE配置包安装的方法，配置运行的方法，则可以避免此类问题。

**解决方法：**

用户明确安装Paddle的python位置，并切换到该python进行安装。可能需要使用python -m pip install paddlepaddle命令确保paddle是安装到该python中。

------

**问题：**MacOS下安装PaddlePaddle后import paddle.fluid出现`Fatal Python error: PyThreadState_Get: no current thread running`错误

**解决方法：**

For Python2.7.x （install by brew): 请使用`export LD_LIBRARY_PATH=/usr/local/Cellar/python@2/2.7.15_1/Frameworks/Python.framework/Versions/2.7 && export DYLD_LIBRARY_PATH=/usr/local/Cellar/python@2/2.7.15_1/Frameworks/Python.framework/Versions/2.7`
For Python2.7.x （install by Python.org): 请使用`export LD_LIBRARY_PATH=/Library/Frameworks/Python.framework/Versions/2.7 && export DYLD_LIBRARY_PATH=/Library/Frameworks/Python.framework/Versions/2.7`
For Python3.5.x （install by Python.org): 请使用`export LD_LIBRARY_PATH=/Library/Frameworks/Python.framework/Versions/3.5/ && export DYLD_LIBRARY_PATH=/Library/Frameworks/Python.framework/Versions/3.5/`


----------
**问题：**GPU安装成功，无法import，是什么原因？

**问题描述：**

使用 `sudo nvidia-docker run --name Paddle -it -v $PWD:/work hub.baidubce.com/paddlepaddle/paddle:latest-gpu-cuda8.0-cudnn7 /bin/bash`，安装成功后，出现如下问题

    import paddle.fluid
    *** Aborted at 1539682149 (unix time) try "date -d @1539682149" if you are using GNU date ***
    PC: @ 0x0 (unknown)
    *** SIGILL (@0x7f6ac6ea9436) received by PID 16 (TID 0x7f6b07bc7700) from PID 18446744072751846454; stack trace: ***

**解决方法：**

请先确定一下机器是否支持AVX2指令集，如果不支持，请按照相应的不支持AVX2指令集的PaddlePaddle，可以解决该问题。


----------


**问题：** 使用PaddlePaddle GPU的Docker镜像时报错， `Cuda Error: CUDA driver version is insufficient for CUDA runtime version`，如何解决？

**报错分析：**

机器上的CUDA驱动偏低导致。

**解决方法：**

需要升级CUDA驱动解决。

- Ubuntu和CentOS环境，需要把相关的驱动和库映射到容器内部。如果使用GPU的docker环境，需要用nvidia-docker来运行，更多请参考nvidia-docker。
-  Windows环境，需要升级CUDA驱动。

------

**问题**： 使用PaddlePaddle时报错，`Error: no CUDA-capable device is detected`，如何解决？

**报错分析：**

CUDA安装错误导致。

**解决方法：**

查找“libcudart.so”所在目录，并将其添加到“LD_LIBRARY_PATH”中。

例如：执行`find / -name libcudart.so`, 发现libcudart.so在“/usr/local/cuda-8.0/targets/x86_64-linux/lib/libcudart.so”路径下， 使用如下命令添加即可。

`export LD_LIBRARY_PATH=/usr/local/cuda-8.0/targets/x86_64-linux/lib/libcudart.so$LD_LIBRARY_PATH`

------

**问题：**如何升级PaddlePaddle？

**答复：**

- GPU环境：

 `pip install -U paddlepaddle-gpu`
或者
`pip install paddlepaddle-gpu == 需要安装的版本号（如2.0）`

- CPU环境：
`pip install -U paddlepaddle`
或者
`pip install paddlepaddle == 需要安装的版本号（如2.0）`

------

**问题：**在GPU上如何选择PaddlePaddle版本？


**答复：**

pip install paddlepaddle-gpu==需要安装的版本号+'.post'+CUDA主版本+CUDNN主版本 例：pip install paddlepaddle-gpu==1.8.4.post97表示需要在CUDA9.0、cuDNN7.0进行安装。更多安装信息请见官网：https://www.paddlepaddle.org.cn/start


----------
**问题：**可以用 IDE 吗？

**答复：**

当然可以，因为源码就在本机上。IDE 默认调用 make 之类的程序来编译源码，我们只需要配置 IDE 来调用 Docker 命令编译源码即可。

很多 PaddlePaddle 开发者使用 Emacs。他们在自己的 `~/.emacs` 配置文件里加两行:

    global-set-key "\C-cc" 'compile
    setq compile-command "docker run --rm -it -v $(git rev-parse --show-toplevel):/paddle paddle:dev"
就可以按 `Ctrl-C` 和 `c` 键来启动编译了。


----------
**问题：**使用docker命令时磁盘不够？

**答复：**

在本文中的例子里，`docker run` 命令里都用了 `--rm` 参数，这样保证运行结束之后的 containers 不会保留在磁盘上。可以用 `docker ps -a` 命令看到停止后但是没有删除的 containers。`docker build` 命令有时候会产生一些中间结果，是没有名字的 images，也会占用磁盘。


----------
**问题：**pip、docker、源码编译安装PaddlePaddle结果有区别吗？

**答复：**区别不大，但三种方法比起来，pip安装更简便一些。


----------
**问题：**Mac系统上NumPy & Python冲突

**问题描述：**

因为需要安装numpy等包，但在Mac自带的Python上无法安装，权限错误导致难以将PaddlePaddle正常安装到Mac本地，改如何解决？

**解决方法：**

Mac上对自带的Python和包有严格的权限保护，最好不要在自带的Python上安装。建议用virtualenv建立一个新的Python环境来操作。virtualenv的基本原理是将机器上的Python运行所需的运行环境完整地拷贝一份。我们可以在一台机器上制造多份拷贝，并在这多个拷贝之间自由切换，这样就相当于在一台机器上拥有了多个相互隔离、互不干扰的Python环境。

下面使用virtualenv为Paddle生成一个专用的Python环境。

安装virtualenv，virtualenv本身也是Python的一个包，可以用pip进行安装：

    sudo -H pip install virtualenv

由于virtualenv需要安装给系统自带的Python，因此需要使用sudo权限。接着使用安装好的virtualenv创建一个新的Python运行环境：

    virtualenv --no-site-packages paddle

`--no-site-packages` 参数表示不拷贝已有的任何第三方包，创造一个完全干净的新Python环境。后面的paddle是我们为这个新创建的环境取的名字。执行完这一步后，当前目录下应该会出现一个名为paddle（或者你取的其他名字）的目录。这个目录里保存了运行一个Python环境所需要的各种文件。 启动运行环境：

    source paddle/bin/activate

执行后会发现命令提示符前面增加了(paddle)字样，说明已经成功启动了名为‘paddle’的Python环境。执行which python，可以发现使用的已经是刚刚创建的paddle目录下的Python。在这个环境中，我们可以自由地进行PaddlePaddle的安装、使用和开发工作，无需担心对系统自带Python的影响。 如果我们经常使用Paddle这个环境，我们每次打开终端后都需要执行一下source paddle/bin/activate来启动环境，比较繁琐。为了简便，可以修改终端的配置文件，来让终端每次启动后自动启动特定的Python环境。 执行:

    vi ~/.bash_profile

打开终端配置文件，并在文件的最后添加一行：

    source paddle/bin/activate

这样，每次打开终端时就会自动启动名为‘paddle’的Python环境了。


----------
**问题：**Python相关单元测试失败

**问题描述：**

安装完了PaddlePaddle后，出现以下python相关的单元测试都过不了的情况：

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

并且查询PaddlePaddle单元测试的日志，提示：

    paddle package is already in your PYTHONPATH. But unittest need a clean environment.
    Please uninstall paddle package before start unittest. Try to 'pip uninstall paddle'.

**解决方法：**

卸载PaddlePaddle包 `pip uninstall paddle`, 清理掉老旧的PaddlePaddle安装包，使得单元测试有一个干净的环境。如果PaddlePaddle包已经在python的site-packages里面，单元测试会引用site-packages里面的python包，而不是源码目录里 `/python` 目录下的python包。同时，即便设置 `PYTHONPATH` 到 `/python` 也没用，因为python的搜索路径是优先已经安装的python包。


----------
**问题：**Windows下支持RTX2060的CUDA功能吗？CUDA支持列表里面看不到是否支持？

**答复：**当前在Windows下RTX2060等显卡目前改篇更新时PaddlePaddle版本为1.8)支持CUDA10和CUDA9，PaddlePaddle的CUDA支持情况可以在官方文档查阅。


----------
**问题：**在Docker镜像上，GPU版本的PaddlePaddle运行结果报错`ERROR: test_cuda (__main__.TestFitALine)`

**问题描述：**

![图片](https://agroup-bos-bj.cdn.bcebos.com/bj-eb74992ee3f1036b2857c76a98f2feebc855f8dc)
![图片](https://agroup-bos-bj.cdn.bcebos.com/bj-490a36c900404826287498c09f8bbf3b52fa49dd)

**解决方法：**

使用`sudo docker run --name paddle-test -v $PWD:/paddle --network=host -it docker.paddlepaddlehub.com/paddle:latest-dev /bin/bash`命令创建的docker容器仅能支持运行CPU版本的PaddlePaddle。 使用如下命令重新开启支持GPU运行的docker容器：

    export CUDA_SO="$(\ls /usr/lib64/libcuda* | xargs -I{} echo '-v {}:{}') $(\ls /usr/lib64/libnvidia* | xargs -I{} echo '-v {}:{}')"
    export DEVICES=$(\ls /dev/nvidia* | xargs -I{} echo '--device {}:{}')
    sudo docker run ${CUDA_SO} ${DEVICES} --rm --name paddle-test-gpu -v /usr/bin/nvidia-smi:/usr/bin/nvidia-smi -v $PWD:/paddle --network=host -it docker.paddlepaddlehub.com/paddle:latest-dev /bin/bash

进入docker之后执行如下命令进行PaddlePaddle的安装及测试运行：

    export LD_LIBRARY_PATH=/usr/lib64:/usr/local/lib:$LD_LIBRARY_PATH
    pip install build/python/dist/*.whl
    python python/paddle/fluid/tests/book/test_fit_a_line.py