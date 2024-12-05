# 安装常见问题

#### 问题：conda 环境下安装 paddlepaddle 3.0.0b0 版本，运行`import paddle`时报错,报错信息为找不到 libpython.so 文件

+ 问题描述：
> ImportError: libpython3.12.so.1.0: cannot open shared object file: No such file or directory

+ 问题分析：
遇到该问题是因为 3.0.0b0 版本中增加了对于 libpython 的依赖，但是使用 conda 安装的 python 环境时，未把 libpython.so 文件所在路径加入到环境变量中，导致找不到该文件。

+ 解决办法：
例如：执行`find / -name libpython3.12.so.1.0`, 发现 libpython 的路径如`/opt/conda/lib/`，使用如下命令安装即可;

```bash
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/conda/lib/
```

##### 问题：使用过程中报找不到 tensorrt 库的日志

+ 问题描述:

> TensorRT dynamic library (libnvinfer.so) that Paddle depends on is not configured correctly. (error code is libnvinfer.so: cannot open shared object file: No such file or directory)
> Suggestions:
> Check if TensorRT is installed correctly and its version is matched with paddlepaddle you installed.
> Configure TensorRT dynamic library environment variables as follows:
> Linux: set LD_LIBRARY_PATH by export LD_LIBRARY_PATH=...
> Windows: set PATH by `set PATH=XXX;
+ 问题分析：

遇到该问题是因为使用的 paddle 默认开始了 TensorRT，但是本地环境中没有找到 TensorRT 的库，该问题只影响使用[Paddle Inference](https://paddleinference.paddlepaddle.org.cn/master/product_introduction/inference_intro.html)开启 TensorRT 预测的场景，对其它方面均不造成影响。

+ 解决办法：

根据提示信息，在环境变量中加入 TensorRT 的库路径。

-----

##### 问题：Windows 环境下，使用 pip install 时速度慢，如何解决？

+ 解决方案：

在 pip 后面加上参数`-i`指定 pip 源，使用国内源获取安装包。

+ 操作步骤：

1. 使用如下命令安装 PaddlePaddle。

   ```bash
   pip3 install paddlepaddle -i https://mirror.baidu.com/pypi/simple/
   ```

你还可以通过如下三个地址获取 pip 安装包，只需修改 `-i` 后网址即可：

1. https://pypi.tuna.tsinghua.edu.cn/simple
2. https://mirrors.aliyun.com/pypi/simple/
3. https://pypi.douban.com/simple/

------

##### 问题：使用 pip install 时报错，`PermissionError: [WinError 5]` ，如何解决？

+ 问题描述：

使用 pip install 时报错，`PermissionError: [WinError 5]` ，

`C:\\Program Files\\python35\\Lib\\site-packages\\graphviz`。

+ 报错分析：

用户权限问题导致，由于用户的 Python 安装到系统文件内（如`Program Files/`），任何的操作都需要管理员权限。

+ 解决方法：

选择“以管理员身份运行”运行 CMD，重新执行安装过程, 使用命令`pip install paddlepaddle`。

------

##### 问题： 使用 pip install 时报错，`ERROR: No matching distribution found for paddlepaddle` ，如何解决？

+ 问题描述：

使用 pip install 时报错，`ERROR: Could not find a version that satisfies the requirement paddlepaddle (from versions: none)`

`ERROR: No matching distribution found for paddlepaddle`

+ 报错分析：

Python 版本不匹配导致。用户使用的是 32 位 Python，但是对应的 32 位 pip 没有 PaddlePaddle 源。

+ 解决方法：

请用户使用 64 位的 Python 进行 PaddlePaddle 安装。

------

##### 问题： 本地使用 import paddle 时报错，`ModuleNotFoundError:No module named ‘paddle’`，如何解决？

+ 报错分析：

原因在于用户的计算机上可能安装了多个版本的 Python，而安装 PaddlePaddle 时的 Python 和`import paddle`时的 Python 版本不一致导致报错。如果用户熟悉 PyCharm 等常见的 IDE 配置包安装的方法，配置运行的方法，则可以避免此类问题。

+ 解决方法：

用户明确安装 Paddle 的 python 位置，并切换到该 python 进行安装。可能需要使用`python -m pip install paddlepaddle`命令确保 paddle 是安装到该 python 中。

------

##### 问题： 使用 PaddlePaddle GPU 的 Docker 镜像时报错， `Cuda Error: CUDA driver version is insufficient for CUDA runtime version`，如何解决？

+ 报错分析：

机器上的 CUDA 驱动偏低导致。

+ 解决方法：

需要升级 CUDA 驱动解决。

1. Ubuntu 和 CentOS 环境，需要把相关的驱动和库映射到容器内部。如果使用 GPU 的 docker 环境，需要用 nvidia-docker 来运行，更多请参考[nvidia-docker](https://github.com/NVIDIA/nvidia-docker)。

2. Windows 环境，需要升级 CUDA 驱动。

------

##### 问题： 使用 PaddlePaddle 时报错，`Error: no CUDA-capable device is detected`，如何解决？

+ 报错分析：

CUDA 安装错误导致。

+ 解决方法：

查找“libcudart.so”所在目录，并将其添加到`LD_LIBRARY_PATH`中。

例如：执行`find / -name libcudart.so`, 发现 libcudart.so 在`/usr/local/cuda-10.0/targets/x86_64-linux/lib/libcudart.so`路径下， 使用如下命令添加即可。

```bash
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/targets/x86_64-linux/lib/libcudart.so:${LD_LIBRARY_PATH}
```

------

##### 问题： 如何升级 PaddlePaddle？

+ 答复：

1. GPU 环境：

  ```bash
  pip install -U paddlepaddle-gpu
  ```

或者

  ```bash
  pip install paddlepaddle-gpu==需要安装的版本号（如 2.0）
  ```

2. CPU 环境：

  ```bash
  pip install -U paddlepaddle
  ```
或者

  ```bash
  pip install paddlepaddle==需要安装的版本号（如 2.0）
  ```

------

##### 问题： 在 GPU 上如何选择 PaddlePaddle 版本？

+ 答复：

首先请确定你本机的 CUDA、cuDNN 版本，飞桨目前 pip 安装适配 CUDA 版本 9.0/10.0/10.1/10.2/11.0，CUDA9.0/10.0/10.1/10.2 配合 cuDNN v7.6.5+，CUDA 工具包 11.0 配合 cuDNN v8.0.4。请确定你安装的是适合的版本。更多安装信息见[官网安装文档](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/2.0/install/pip/windows-pip.html)

------

##### 问题： import paddle 报错, dlopen: cannot load any more object with static TLS, 如何解决？

+ 答复：

glibc 版本过低，建议使用官方提供的 docker 镜像或者将 glibc 升级到 2.23+。

------

##### 问题： python2.7 中，如果使用 Paddle1.8.5 之前的版本，import paddle 时，报错，提示`/xxxx/rarfile.py, line820, print(f.filename, file=file), SyntaxError: invalid syntax`，如何解决？

+ 答复：

rarfile 版本太高，它的最新版本已经不支持 python2.x 了，可以通过`pip install rarfile==3.0`安装 3.0 版本的 rarfile 即可。
