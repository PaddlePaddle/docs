# 安装常见问题


##### 问题：使用过程中报找不到tensorrt库的日志

+ 问题描述:

> TensorRT dynamic library (libnvinfer.so) that Paddle depends on is not configured correctly. (error code is libnvinfer.so: cannot open shared object file: No such file or directory)  
> Suggestions:  
> Check if TensorRT is installed correctly and its version is matched with paddlepaddle you installed.  
> Configure TensorRT dynamic library environment variables as follows:  
> Linux: set LD_LIBRARY_PATH by export LD_LIBRARY_PATH=...  
> Windows: set PATH by `set PATH=XXX;  
+ 问题分析：

遇到该问题是因为使用的paddle默认开始了TensorRT，但是本地环境中没有找到TensorRT的库，该问题只影响使用[Paddle Inference](https://paddleinference.paddlepaddle.org.cn/master/product_introduction/inference_intro.html)开启TensorRT预测的场景，对其它方面均不造成影响。

+ 解决办法：

根据提示信息，在环境变量中加入TensorRT的库路径。

-----

##### 问题：Windows环境下，使用pip install时速度慢，如何解决？

+ 解决方案：

在pip后面加上参数`-i`指定pip源，使用国内源获取安装包。

+ 操作步骤：

1. Python2情况下，使用如下命令安装PaddlePaddle。

   ```bash
   pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple/
   ```

2. Python3情况下，使用如下命令安装PaddlePaddle。

   ```bash
   pip3 install paddlepaddle -i https://mirror.baidu.com/pypi/simple/
   ```

你还可以通过如下三个地址获取pip安装包，只需修改 `-i` 后网址即可：

1. https://pypi.tuna.tsinghua.edu.cn/simple
2. https://mirrors.aliyun.com/pypi/simple/
3. https://pypi.douban.com/simple/

------

##### 问题：使用pip install时报错，`PermissionError: [WinError 5]` ，如何解决？

+ 问题描述：

使用pip install时报错，`PermissionError: [WinError 5]` ，

`C:\\Program Files\\python35\\Lib\\site-packages\\graphviz`。

+ 报错分析：

用户权限问题导致，由于用户的Python安装到系统文件内（如`Program Files/`），任何的操作都需要管理员权限。

+ 解决方法：

选择“以管理员身份运行”运行CMD，重新执行安装过程, 使用命令`pip install paddlepaddle`。

------

##### 问题： 使用pip install时报错，`ERROR: No matching distribution found for paddlepaddle` ，如何解决？

+ 问题描述：

使用pip install时报错，`ERROR: Could not find a version that satisfies the requirement paddlepaddle (from versions: none)`

`ERROR: No matching distribution found for paddlepaddle`

<img src="https://agroup-bos-bj.cdn.bcebos.com/bj-febb18fb78004dc17f18d60a009dc6a8bd907251" alt="图片" />

+ 报错分析：

Python版本不匹配导致。用户使用的是32位Python，但是对应的32位pip没有PaddlePaddle源。

+ 解决方法：

请用户使用64位的Python进行PaddlePaddle安装。

------

##### 问题： 本地使用import paddle时报错，`ModuleNotFoundError:No module named ‘paddle’`，如何解决？

+ 报错分析：

原因在于用户的计算机上可能安装了多个版本的Python，而安装PaddlePaddle时的Python和`import paddle`时的Python版本不一致导致报错。如果用户熟悉PyCharm等常见的IDE配置包安装的方法，配置运行的方法，则可以避免此类问题。

+ 解决方法：

用户明确安装Paddle的python位置，并切换到该python进行安装。可能需要使用`python -m pip install paddlepaddle`命令确保paddle是安装到该python中。

------

##### 问题： 使用PaddlePaddle GPU的Docker镜像时报错， `Cuda Error: CUDA driver version is insufficient for CUDA runtime version`，如何解决？

+ 报错分析：

机器上的CUDA驱动偏低导致。

+ 解决方法：

需要升级CUDA驱动解决。

1. Ubuntu和CentOS环境，需要把相关的驱动和库映射到容器内部。如果使用GPU的docker环境，需要用nvidia-docker来运行，更多请参考[nvidia-docker](https://github.com/NVIDIA/nvidia-docker)。

2. Windows环境，需要升级CUDA驱动。

------

##### 问题： 使用PaddlePaddle时报错，`Error: no CUDA-capable device is detected`，如何解决？

+ 报错分析：

CUDA安装错误导致。

+ 解决方法：

查找“libcudart.so”所在目录，并将其添加到`LD_LIBRARY_PATH`中。

例如：执行`find / -name libcudart.so`, 发现libcudart.so在`/usr/local/cuda-10.0/targets/x86_64-linux/lib/libcudart.so`路径下， 使用如下命令添加即可。

```bash
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/targets/x86_64-linux/lib/libcudart.so:${LD_LIBRARY_PATH}
```

------

##### 问题： 如何升级PaddlePaddle？

+ 答复：

1. GPU环境：

  ```bash
  pip install -U paddlepaddle-gpu
  ```

或者

  ```bash
  pip install paddlepaddle-gpu==需要安装的版本号（如2.0）
  ```

2. CPU环境：

  ```bash
  pip install -U paddlepaddle
  ```
或者

  ```bash
  pip install paddlepaddle==需要安装的版本号（如2.0）
  ```

------

##### 问题： 在GPU上如何选择PaddlePaddle版本？

+ 答复：

首先请确定你本机的CUDA、cuDNN版本，飞桨目前pip安装适配CUDA版本9.0/10.0/10.1/10.2/11.0，CUDA9.0/10.0/10.1/10.2 配合 cuDNN v7.6.5+，CUDA 工具包11.0配合cuDNN v8.0.4。请确定你安装的是适合的版本。更多安装信息见[官网安装文档](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/2.0/install/pip/windows-pip.html)

------

##### 问题： import paddle报错, dlopen: cannot load any more object with static TLS, 如何解决？

+ 答复：

glibc版本过低，建议使用官方提供的docker镜像或者将glibc升级到2.23+。

------

##### 问题： python2.7中，如果使用Paddle1.8.5之前的版本，import paddle时，报错，提示`/xxxx/rarfile.py, line820, print(f.filename, file=file), SyntaxError: invalid syntax`，如何解决？

+ 答复：

rarfile版本太高，它的最新版本已经不支持python2.x了，可以通过`pip install rarfile==3.0`安装3.0版本的rarfile即可。
