# 安装类FAQ

##### 问题：Windows环境下，使用pip install时速度慢，如何解决？

+ 解决方案：

在pip后面加上参数`-i`指定pip源，使用国内源获取安装包。

+ 操作步骤：

1. Python2情况下，使用如下命令安装PaddlePaddle。

   `pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple/`

2. Python3情况下，使用如下命令安装PaddlePaddle。

   `pip3 install paddlepaddle -i https://mirror.baidu.com/pypi/simple/`

您还可以通过如下三个地址获取pip安装包，只需修改 `-i` 后网址即可：

https://pypi.tuna.tsinghua.edu.cn/simple
https://mirrors.aliyun.com/pypi/simple/
https://pypi.douban.com/simple/

------

##### 问题：使用pip install时报错，`PermissionError: [WinError 5]` ，如何解决？

+ 问题描述：

使用pip install时报错，`PermissionError: [WinError 5]` ，

`C:\\program fiels\\python35\\Lib\\site-packages\\graphviz`。

+ 报错分析：

用户权限问题导致，由于用户的Python安装到系统文件内（如”Program Files/“），任何的操作都需要管理员权限。

+ 解决方法：

选择“以管理员身份运行”运行CMD，重新执行安装过程, 使用命令sudo pip install paddlepaddle

------

##### 问题： 使用pip install时报错，`ERROR: No matching distribution found for paddlepaddle` ，如何解决？

+ 问题描述：

使用pip install时报错，`ERROR: Could not find a version that satisfies the requirement paddlepaddle (from versions: none)`

``ERROR: No matching distribution found for paddlepaddle`
![图片](https://agroup-bos-bj.cdn.bcebos.com/bj-febb18fb78004dc17f18d60a009dc6a8bd907251)

+ 报错分析：

Python版本不匹配导致。用户使用的是32位Python，但是对应的32位pip没有PaddlePaddle源。

+ 解决方法：

请用户使用64位的Python进行PaddlePaddle安装。

------

##### 问题： 在GPU上执行程序报错，`Error：Segmentation fault`，如何解决？

+ 问题描述：

在GPU版本为`paddlepaddle_gpu-1.8.4.post87-cp27-cp27mu-manylinux1_x86_64.whl`的环境上执行一个程序，出现`Error：Segmentation fault`。如果将`place`修改“cpu”，则程序可正常运行。

+ 报错分析：

造成该报错的原因通常是环境不匹配导致的。安装时，GPU版本为`paddlepaddle_gpu-1.8.4.post87-cp27-cp27mu-manylinux1_x86_64.whl`，`post87`表示需要在CUDA8.0、cuDNN7.0进行编译。如果机器上没有安装对应版本的CUDA和cuDNN，会导致执行程序时报错。

此外值得注意的是，配置PaddlePaddle的GPU版本，不仅需要CUDA和cuDNN版本匹配，还需要与PaddlePaddle版本匹配。出现类似错误时请检查这三个程序的版本是否匹配。

+ 解决方法：

CUDA的安装可参考：https://docs.nvidia.com/cuda/archive/10.0/index.html；cuDNN的安装可参考：https://docs.nvidia.com/deeplearning/cudnn/install-guide/#install-windows。

------

##### 问题： 本地使用import paddle时报错，`ModuleNotFoundError:No module named ‘paddle’`，如何解决？

+ 报错分析：

原因在于用户的计算机上可能安装了多个版本的Python，而安装PaddlePaddle时的Python和import paddle时的Python版本不一致导致报错。如果用户熟悉PyCharm等常见的IDE配置包安装的方法，配置运行的方法，则可以避免此类问题。

+ 解决方法：

用户明确安装Paddle的python位置，并切换到该python进行安装。可能需要使用python -m pip install paddlepaddle命令确保paddle是安装到该python中。

------

##### 问题： 使用PaddlePaddle GPU的Docker镜像时报错， `Cuda Error: CUDA driver version is insufficient for CUDA runtime version`，如何解决？

+ 报错分析：

机器上的CUDA驱动偏低导致。

+ 解决方法：

需要升级CUDA驱动解决。

1. Ubuntu和CentOS环境，需要把相关的驱动和库映射到容器内部。如果使用GPU的docker环境，需要用nvidia-docker来运行，更多请参考nvidia-docker。

2. Windows环境，需要升级CUDA驱动。

------

##### 问题： 使用PaddlePaddle时报错，`Error: no CUDA-capable device is detected`，如何解决？

+ 报错分析：

CUDA安装错误导致。

+ 解决方法：

查找“libcudart.so”所在目录，并将其添加到“LD_LIBRARY_PATH”中。

例如：执行`find / -name libcudart.so`, 发现libcudart.so在“/usr/local/cuda-10.0/targets/x86_64-linux/lib/libcudart.so”路径下， 使用如下命令添加即可。

`export LD_LIBRARY_PATH=/usr/local/cuda-10.0/targets/x86_64-linux/lib/libcudart.so$LD_LIBRARY_PATH`

------

##### 问题： 如何升级PaddlePaddle？

+ 答复：

1. GPU环境：


 `pip install -U paddlepaddle-gpu`

或者

`pip install paddlepaddle-gpu == 需要安装的版本号（如2.0）`

2. CPU环境：

`pip install -U paddlepaddle`

或者

`pip install paddlepaddle == 需要安装的版本号（如2.0）`

------

##### 问题： 在GPU上如何选择PaddlePaddle版本？

+ 答复：

pip install paddlepaddle-gpu==需要安装的版本号+'.post'+CUDA主版本+CUDNN主版本 例：pip install paddlepaddle-gpu==1.8.4.post97表示需要在CUDA9.0、cuDNN7.0进行安装。更多安装信息请见官网：https://www.paddlepaddle.org.cn/start
