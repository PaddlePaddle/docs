<a name="third_party"></a>
# 附录

<a name="gpu-install"></a>

## **飞桨支持的 Nvidia GPU 架构及安装方式**
<p align="center">
<table>
    <thead>
    <tr>
        <th> GPU 架构 </th>
        <th> Compute Capability </th>
        <th> 对应 GPU 硬件型号 </th>
        <th> 请下载以下 CUDA 版本的飞桨安装包 </th>
    </tr>
    </thead>
    <tbody>
    <tr>
        <td> Pascal </td>
        <td> sm_60 </td>
        <td> Quadro GP100, Tesla P100, DGX-1 </td>
        <td> CUDA10、CUDA11 </td>
    </tr>
    <tr>
        <td> Pascal </td>
        <td> sm_61 </td>
        <td> GTX 1080, GTX 1070, GTX 1060, GTX 1050, GTX 1030 (GP108), GT 1010 (GP108) Titan Xp, Tesla P40, Tesla P4 </td>
        <td> CUDA10、CUDA11 </td>
    </tr>
    <tr>
        <td> Volta </td>
        <td> sm_70 </td>
        <td> DGX-1 with Volta, Tesla V100, GTX 1180 (GV104), Titan V, Quadro GV100 </td>
        <td> CUDA10、CUDA11 </td>
    </tr>
    <tr>
        <td> Turing </td>
        <td> sm_75 </td>
        <td> GTX/RTX Turing – GTX 1660 Ti, RTX 2060, RTX 2070, RTX 2080, Titan RTX, Quadro RTX 4000, Quadro RTX 5000, Quadro RTX 6000, Quadro RTX 8000, Quadro T1000/T2000, Tesla T4 </td>
        <td> CUDA10、CUDA11 </td>
    </tr>
    <tr>
        <td> Ampere </td>
        <td> sm_80 </td>
        <td> NVIDIA A100, GA100, NVIDIA DGX-A100 </td>
        <td> CUDA11 </td>
    </tr>
    <tr>
        <td> Ampere </td>
        <td> sm_86 </td>
        <td> Tesla GA10x cards, RTX Ampere – RTX 3080, GA102 – RTX 3090, RTX A2000, A3000, RTX A4000, A5000, A6000, NVIDIA A40, GA106 – RTX 3060, GA104 – RTX 3070, GA107 – RTX 3050, RTX A10, RTX A16, RTX A40, A2 Tensor Core GPU </td>
        <td> CUDA11、CUDA11.2（推荐） </td>
    </tr>
    <tr>
        <td> Hopper </td>
        <td> sm_90 </td>
        <td> NVIDIA H100, H800 </td>
        <td> CUDA12 </td>
    </tr>
    </tbody>
</table>
</p>

</br></br>

## **编译依赖表**

<p align="center">
<table>
    <thead>
    <tr>
        <th> 依赖包名称 </th>
        <th> 版本 </th>
        <th> 说明 </th>
        <th> 安装命令 </th>
    </tr>
    </thead>
    <tbody>
    <tr>
        <td> CMake </td>
        <td> 3.18, 3.19(推荐) </td>
        <td>  </td>
        <td>  </td>
    </tr>
    <tr>
        <td> GCC (Linux Only) </td>
        <td> 8.2 / 12.2 </td>
        <td>  推荐使用 CentOS 的 devtools2 </td>
        <td>  </td>
    </tr>
    <tr>
        <td> Clang (MacOS Only) </td>
        <td> 9.0 及以上 </td>
        <td> 通常使用 MacOS 10.11 及以上的系统对应的 Clang 版本即可 </td>
        <td>  </td>
    </tr>
        <tr>
        <td> Python（64 bit） </td>
        <td> 3.8+.x </td>
        <td> 依赖 libpython3.8+.so </td>
        <td> 请访问<a href="https://www.python.org">Python 官网</a></td>
    </tr>
    <tr>
        <td> SWIG </td>
        <td> 最低 2.0 </td>
        <td>  </td>
        <td> <code>apt install swig </code> 或 <code> yum install swig </code> </td>
    </tr>
    <tr>
        <td> wget </td>
        <td> any </td>
        <td>  </td>
        <td> <code> apt install wget </code>  或 <code> yum install wget </code> </td>
    </tr>
    <tr>
        <td> openblas </td>
        <td> any </td>
        <td> 可选 </td>
        <td>  </td>
    </tr>
    <tr>
        <td> pip </td>
        <td> >=20.2.2 </td>
        <td>  </td>
        <td> <code> apt install python-pip </code> 或 <code> yum install python-pip </code> </td>
    </tr>
    <tr>
        <td> numpy </td>
        <td> >=1.13.0 </td>
        <td>  </td>
        <td> <code> pip install numpy </code> </td>
    </tr>
    <tr>
        <td> protobuf </td>
        <td> >=3.20.2 </td>
        <td>  </td>
        <td> <code> pip install protobuf </code> </td>
    </tr>
    <tr>
        <td> wheel </td>
        <td> any </td>
        <td>  </td>
        <td> <code> pip install wheel </code> </td>
    </tr>
    <tr>
        <td> patchELF </td>
        <td> any </td>
        <td>  </td>
        <td> <code> apt install patchelf </code> 或参见 github <a href="https://gist.github.com/ruario/80fefd174b3395d34c14">patchELF 官方文档</a></td>
    </tr>
    <tr>
        <td> go </td>
        <td> >=1.8 </td>
        <td> 可选 </td>
        <td>  </td>
    </tr>
    <tr>
        <td> setuptools </td>
        <td> >= 50.3.2 </td>
        <td> </td>
        <td>  </td>
    </tr>
    <tr>
        <td> unrar </td>
        <td>  </td>
        <td> </td>
        <td> brew install unrar (For MacOS), apt-get install unrar (For Ubuntu) </td>
    </tr>
    </tbody>
</table>
</p>


<a name="Compile"></a>
</br></br>
## **编译选项表**

<p align="center">
<table>
    <thead>
    <tr>
        <th> 选项 </th>
        <th> 说明 </th>
        <th> 默认值 </th>
    </tr>
    </thead>
    <tbody>
    <tr>
        <td> WITH_GPU </td>
        <td> 是否支持 CUDA </td>
        <td> ON </td>
    </tr>
    <tr>
        <td> WITH_ROCM </td>
        <td> 是否支持 ROCM </td>
        <td> OFF </td>
    </tr>
    <tr>
        <td> WITH_AVX </td>
        <td> 是否编译含有 AVX 指令集的 PaddlePaddle 二进制文件 </td>
        <td> ON </td>
    </tr>
    <tr>
        <td> WITH_PYTHON </td>
        <td> 是否内嵌 PYTHON 解释器 </td>
        <td> ON </td>
    </tr>
    <tr>
        <td> WITH_TESTING </td>
        <td> 是否开启单元测试 </td>
        <td> OFF </td>
    </tr>
    <tr>
        <td> WITH_MKL </td>
        <td> 是否使用 MKL 数学库，如果为否则是用 OpenBLAS </td>
        <td> ON </td>
    </tr>
    <tr>
        <td> WITH_SYSTEM_BLAS </td>
        <td> 是否使用系统自带的 BLAS </td>
        <td> OFF </td>
    </tr>
    <tr>
        <td> WITH_DISTRIBUTE </td>
        <td> 是否编译带有分布式的版本 </td>
        <td> OFF </td>
    </tr>
    <tr>
        <td> WITH_BRPC_RDMA </td>
        <td> 是否使用 BRPC RDMA 作为 RPC 协议 </td>
        <td> OFF </td>
    </tr>
        <tr>
        <td> ON_INFER </td>
        <td> 是否打开预测优化 </td>
        <td> OFF </td>
    </tr>
    <tr>
        <tr>
        <td> CUDA_ARCH_NAME </td>
        <td> 是否只针对当前 CUDA 架构编译 </td>
        <td> All:编译所有可支持的 CUDA 架构 可选：Auto 自动识别当前环境的架构编译 </td>
    </tr>
    <tr>
        <tr>
        <td> TENSORRT_ROOT </td>
        <td> 指定 TensorRT 路径 </td>
        <td> Windows 下默认值为'/'，Linux 下默认值为 '/usr/' </td>
    </tr>
   </tbody>
</table>
</p>





**BLAS**

PaddlePaddle 支持 [MKL](https://software.intel.com/en-us/mkl) 和 [OpenBlAS](http://www.openblas.net) 两种 BLAS 库。默认使用 MKL。如果使用 MKL 并且机器含有 AVX2 指令集，还会下载 MKL-DNN 数学库，详细参考[这里](https://github.com/PaddlePaddle/Paddle/tree/release/0.11.0/doc/design/mkldnn#cmake) 。

如果关闭 MKL，则会使用 OpenBLAS 作为 BLAS 库。

**CUDA/cuDNN**

PaddlePaddle 在编译时/运行时会自动找到系统中安装的 CUDA 和 cuDNN 库进行编译和执行。 使用参数 `-DCUDA_ARCH_NAME=Auto` 可以指定开启自动检测 SM 架构，加速编译。

PaddlePaddle 可以使用 cuDNN v5.1 之后的任何一个版本来编译运行，但尽量请保持编译和运行使用的 cuDNN 是同一个版本。 我们推荐使用最新版本的 cuDNN。

**编译选项的设置**

PaddePaddle 通过编译时指定路径来实现引用各种 BLAS/CUDA/cuDNN 库。cmake 编译时，首先在系统路径（ `/usr/lib` 和 `/usr/local/lib` ）中搜索这几个库，同时也会读取相关路径变量来进行搜索。 通过使用`-D`命令可以设置，例如：

> `cmake .. -DWITH_GPU=ON -DWITH_TESTING=OFF -DCUDNN_ROOT=/opt/cudnnv5`

**注意**：这几个编译选项的设置，只在第一次 cmake 的时候有效。如果之后想要重新设置，推荐清理整个编译目录（ rm -rf ）后，再指定。


<a name="whls"></a>
</br></br>
## **安装包列表**

<p align="center">
<table>
    <thead>
    <tr>
        <th> 版本号 </th>
        <th> 版本说明 </th>
    </tr>
    </thead>
    <tbody>
    <tr>
        <td> paddlepaddle==[版本号] 例如 paddlepaddle==2.6.0 </td>
        <td> 只支持 CPU 对应版本的 PaddlePaddle，具体版本请参见<a href=https://pypi.org/project/paddlepaddle/#history>Pypi</a> </td>
    </tr>
    <tr>
        <td> paddlepaddle-gpu==[版本号] 例如 paddlepaddle-gpu==2.6.0 </td>
        <td> 默认安装支持 CUDA 11.8 和 cuDNN 8 的对应[版本号]的 PaddlePaddle 安装包 </td>
    </tr>
   </tbody>
</table>
</p>

您可以在 [Release History](https://pypi.org/project/paddlepaddle-gpu/#history) 中找到 PaddlePaddle-gpu 的各个发行版本。
> 其中`postXX` 对应的是 CUDA 和 cuDNN 的版本，`postXX`之前的数字代表 Paddle 的版本

需要注意的是，命令中<code> paddlepaddle-gpu==2.6.0 </code> 在 windows 环境下，会默认安装支持 CUDA 11.8 和 cuDNN 8 的对应[版本号]的 PaddlePaddle 安装包

<a name="ciwhls-release"></a>
</br></br>

## **多版本 whl 包列表-Release**

<p align="center">
<table>
    <thead>
    <tr>
        <th> 版本说明 </th>
        <th> cp38-cp38    </th>
        <th> cp39-cp39    </th>
        <th> cp310-cp310    </th>
        <th> cp311-cp311    </th>
        <th> cp312-cp312    </th>
    </tr>
    </thead>
    <tbody>
    <tr>
        <td> cpu-mkl-avx </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.6.0/linux/linux-cpu-mkl-avx/paddlepaddle-2.6.0-cp38-cp38-linux_x86_64.whl"> paddlepaddle-2.6.0-cp38-cp38-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.6.0/linux/linux-cpu-mkl-avx/paddlepaddle-2.6.0-cp39-cp39-linux_x86_64.whl"> paddlepaddle-2.6.0-cp39-cp39-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.6.0/linux/linux-cpu-mkl-avx/paddlepaddle-2.6.0-cp310-cp310-linux_x86_64.whl"> paddlepaddle-2.6.0-cp310-cp310-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.6.0/linux/linux-cpu-mkl-avx/paddlepaddle-2.6.0-cp311-cp311-linux_x86_64.whl"> paddlepaddle-2.6.0-cp311-cp311-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.6.0/linux/linux-cpu-mkl-avx/paddlepaddle-2.6.0-cp312-cp312-linux_x86_64.whl"> paddlepaddle-2.6.0-cp312-cp312-linux_x86_64.whl</a></td>
    </tr>
    <tr>
        <td> cpu-openblas-avx </td>
        <td> - </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.6.0/linux/linux-cpu-openblas-avx/paddlepaddle-2.6.0-cp38-cp38-linux_x86_64.whl"> paddlepaddle-2.6.0-cp38-cp38-linux_x86_64.whl</a></td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
    </tr>
    <tr>
        <td> cuda11.2-cudnn8.1-mkl-gcc8.2-avx </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.6.0/linux/linux-gpu-cuda11.2-cudnn8-mkl-gcc8.2-avx/paddlepaddle_gpu-2.6.0.post112-cp38-cp38-linux_x86_64.whl">
        paddlepaddle_gpu-2.6.0.post112-cp38-cp38-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.6.0/linux/linux-gpu-cuda11.2-cudnn8-mkl-gcc8.2-avx/paddlepaddle_gpu-2.6.0.post112-cp39-cp39-linux_x86_64.whl">
        paddlepaddle_gpu-2.6.0.post112-cp39-cp39-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.6.0/linux/linux-gpu-cuda11.2-cudnn8-mkl-gcc8.2-avx/paddlepaddle_gpu-2.6.0.post112-cp310-cp310-linux_x86_64.whl">
        paddlepaddle_gpu-2.6.0.post112-cp310-cp310-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.6.0/linux/linux-gpu-cuda11.2-cudnn8-mkl-gcc8.2-avx/paddlepaddle_gpu-2.6.0.post112-cp311-cp311-linux_x86_64.whl">
        paddlepaddle_gpu-2.6.0.post112-cp311-cp311-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.6.0/linux/linux-gpu-cuda11.2-cudnn8-mkl-gcc8.2-avx/paddlepaddle_gpu-2.6.0.post112-cp312-cp312-linux_x86_64.whl">
        paddlepaddle_gpu-2.6.0.post112-cp312-cp312-linux_x86_64.whl</a></td>
    </tr>
    <tr>
        <td> cuda11.6-cudnn8.4-mkl-gcc8.2-avx </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.6.0/linux/linux-gpu-cuda11.6-cudnn8.4.0-mkl-gcc8.2-avx/paddlepaddle_gpu-2.6.0.post116-cp38-cp38-linux_x86_64.whl">
        paddlepaddle_gpu-2.6.0.post116-cp38-cp38-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.6.0/linux/linux-gpu-cuda11.6-cudnn8.4.0-mkl-gcc8.2-avx/paddlepaddle_gpu-2.6.0.post116-cp39-cp39-linux_x86_64.whl">
        paddlepaddle_gpu-2.6.0.post116-cp39-cp39-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.6.0/linux/linux-gpu-cuda11.6-cudnn8.4.0-mkl-gcc8.2-avx/paddlepaddle_gpu-2.6.0.post116-cp310-cp310-linux_x86_64.whl">
        paddlepaddle_gpu-2.6.0.post116-cp310-cp310-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.6.0/linux/linux-gpu-cuda11.6-cudnn8.4.0-mkl-gcc8.2-avx/paddlepaddle_gpu-2.6.0.post116-cp311-cp311-linux_x86_64.whl">
        paddlepaddle_gpu-2.6.0.post116-cp311-cp311-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.6.0/linux/linux-gpu-cuda11.6-cudnn8.4.0-mkl-gcc8.2-avx/paddlepaddle_gpu-2.6.0.post116-cp321-cp312-linux_x86_64.whl">
        paddlepaddle_gpu-2.6.0.post116-cp312-cp312-linux_x86_64.whl</a></td>
    </tr>
    <tr>
        <td> cuda11.7-cudnn8.4-mkl-gcc8.2-avx </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.6.0/linux/linux-gpu-cuda11.7-cudnn8.4.1-mkl-gcc8.2-avx/paddlepaddle_gpu-2.6.0.post117-cp38-cp38-linux_x86_64.whl">
        paddlepaddle_gpu-2.6.0.post117-cp38-cp38-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.6.0/linux/linux-gpu-cuda11.7-cudnn8.4.1-mkl-gcc8.2-avx/paddlepaddle_gpu-2.6.0.post117-cp39-cp39-linux_x86_64.whl">
        paddlepaddle_gpu-2.6.0.post117-cp39-cp39-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.6.0/linux/linux-gpu-cuda11.7-cudnn8.4.1-mkl-gcc8.2-avx/paddlepaddle_gpu-2.6.0.post117-cp310-cp310-linux_x86_64.whl">
        paddlepaddle_gpu-2.6.0.post117-cp310-cp310-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.6.0/linux/linux-gpu-cuda11.7-cudnn8.4.1-mkl-gcc8.2-avx/paddlepaddle_gpu-2.6.0.post117-cp311-cp311-linux_x86_64.whl">
        paddlepaddle_gpu-2.6.0.post117-cp310-cp310-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.6.0/linux/linux-gpu-cuda11.7-cudnn8.4.1-mkl-gcc8.2-avx/paddlepaddle_gpu-2.6.0.post117-cp312-cp312-linux_x86_64.whl">
        paddlepaddle_gpu-2.6.0.post117-cp310-cp310-linux_x86_64.whl</a></td>
    </tr>
    <tr>
        <td> cuda11.8-cudnn8.6-mkl-gcc8.2-avx </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.6.0/linux/linux-gpu-cuda11.8-cudnn8.6-mkl-gcc8.2-avx/paddlepaddle_gpu-2.6.0-cp38-cp38-linux_x86_64.whl">
        paddlepaddle_gpu-2.6.0-cp38-cp38-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.6.0/linux/linux-gpu-cuda11.8-cudnn8.6-mkl-gcc8.2-avx/paddlepaddle_gpu-2.6.0-cp39-cp39-linux_x86_64.whl">
        paddlepaddle_gpu-2.6.0-cp39-cp39-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.6.0/linux/linux-gpu-cuda11.8-cudnn8.6-mkl-gcc8.2-avx/paddlepaddle_gpu-2.6.0-cp310-cp310-linux_x86_64.whl">
        paddlepaddle_gpu-2.6.0-cp310-cp310-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.6.0/linux/linux-gpu-cuda11.8-cudnn8.6-mkl-gcc8.2-avx/paddlepaddle_gpu-2.6.0-cp311-cp311-linux_x86_64.whl">
        paddlepaddle_gpu-2.6.0-cp311-cp311-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.6.0/linux/linux-gpu-cuda11.8-cudnn8.6-mkl-gcc8.2-avx/paddlepaddle_gpu-2.6.0-cp312-cp312-linux_x86_64.whl">
        paddlepaddle_gpu-2.6.0-cp312-cp312-linux_x86_64.whl</a></td>
    </tr>
        <td> cuda12.0-cudnn8.9-mkl-gcc12.2-avx </td>
        <td> - </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.6.0/linux/linux-gpu-cuda12.0-cudnn8.9-mkl-gcc12.2-avx/paddlepaddle_gpu-2.6.0.post120-cp38-cp38-linux_x86_64.whl">
        paddlepaddle_gpu-2.6.0.post120-cp38-cp38-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.6.0/linux/linux-gpu-cuda12.0-cudnn8.9-mkl-gcc12.2-avx/paddlepaddle_gpu-2.6.0.post120-cp39-cp39-linux_x86_64.whl">
        paddlepaddle_gpu-2.6.0.post120-cp39-cp39-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.6.0/linux/linux-gpu-cuda12.0-cudnn8.9-mkl-gcc12.2-avx/paddlepaddle_gpu-2.6.0.post120-cp310-cp310-linux_x86_64.whl">
        paddlepaddle_gpu-2.6.0.post120-cp310-cp310-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.6.0/linux/linux-gpu-cuda12.0-cudnn8.9-mkl-gcc12.2-avx/paddlepaddle_gpu-2.6.0.post120-cp311-cp311-linux_x86_64.whl">
        paddlepaddle_gpu-2.6.0.post120-cp311-cp311-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.6.0/linux/linux-gpu-cuda12.0-cudnn8.9-mkl-gcc12.2-avx/paddlepaddle_gpu-2.6.0.post120-cp312-cp312-linux_x86_64.whl">
        paddlepaddle_gpu-2.6.0.post120-cp312-cp312-linux_x86_64.whl</a></td>
    </tr>
    <tr>
        <td> macos-cpu-openblas </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.6.0/macos/macos-cpu-openblas/paddlepaddle-2.6.0-cp38-cp38-macosx_10_9_x86_64.whl">
        paddlepaddle-2.6.0-cp38-cp38-macosx_10_14_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.6.0/macos/macos-cpu-openblas/paddlepaddle-2.6.0-cp39-cp39-macosx_10_9_x86_64.whl">
        paddlepaddle-2.6.0-cp39-cp39-macosx_10_14_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.6.0/macos/macos-cpu-openblas/paddlepaddle-2.6.0-cp310-cp310-macosx_10_9_x86_64.whl">
        paddlepaddle-2.6.0-cp310-cp310-macosx_10_14_universal2.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.6.0/macos/macos-cpu-openblas/paddlepaddle-2.6.0-cp311-cp311-macosx_10_9_x86_64.whl">
        paddlepaddle-2.6.0-cp311-cp311-macosx_10_14_universal2.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.6.0/macos/macos-cpu-openblas/paddlepaddle-2.6.0-cp312-cp312-macosx_10_9_x86_64.whl">
        paddlepaddle-2.6.0-cp312-cp312-macosx_10_14_universal2.whl</a></td>
    </tr>
    <tr>
        <td> macos-cpu-openblas-m1 </td>
        <td> - </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.6.0/macos/macos-cpu-openblas-m1/paddlepaddle-2.6.0-cp38-cp38-macosx_11_0_arm64.whl">
        paddlepaddle-2.6.0-cp38-cp38-macosx_11_0_arm64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.6.0/macos/macos-cpu-openblas-m1/paddlepaddle-2.6.0-cp39-cp39-macosx_11_0_arm64.whl">
        paddlepaddle-2.6.0-cp39-cp39-macosx_11_0_arm64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.6.0/macos/macos-cpu-openblas-m1/paddlepaddle-2.6.0-cp310-cp310-macosx_11_0_arm64.whl">
        paddlepaddle-2.6.0-cp310-cp310-macosx_11_0_arm64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.6.0/macos/macos-cpu-openblas-m1/paddlepaddle-2.6.0-cp311-cp311-macosx_11_0_arm64.whl">
        paddlepaddle-2.6.0-cp311-cp311-macosx_11_0_arm64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.6.0/macos/macos-cpu-openblas-m1/paddlepaddle-2.6.0-cp312-cp312-macosx_11_0_arm64.whl">
        paddlepaddle-2.6.0-cp312-cp312-macosx_11_0_arm64.whl</a></td>
    </tr>
    <tr>
        <td> win-cpu-mkl-avx </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.6.0/windows/windows-cpu-avx-mkl-vs2017/paddlepaddle-2.6.0-cp38-cp38-win_amd64.whl"> paddlepaddle-2.6.0-cp38-cp38-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.6.0/windows/windows-cpu-avx-mkl-vs2017/paddlepaddle-2.6.0-cp39-cp39-win_amd64.whl"> paddlepaddle-2.6.0-cp39-cp39-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.6.0/windows/windows-cpu-avx-mkl-vs2017/paddlepaddle-2.6.0-cp310-cp310-win_amd64.whl"> paddlepaddle-2.6.0-cp310-cp310-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.6.0/windows/windows-cpu-avx-mkl-vs2017/paddlepaddle-2.6.0-cp311-cp311-win_amd64.whl"> paddlepaddle-2.6.0-cp311-cp311-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.6.0/windows/windows-cpu-avx-mkl-vs2017/paddlepaddle-2.6.0-cp312-cp312-win_amd64.whl"> paddlepaddle-2.6.0-cp312-cp312-win_amd64.whl</a></td>
    </tr>
    <tr>
        <td> win-cpu-openblas-avx </td>
        <td> - </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.6.0/windows/windows-cpu-avx-openblas-vs2017/paddlepaddle-2.6.0-cp38-cp38-win_amd64.whl"> paddlepaddle-2.6.0-cp38-cp38-win_amd64.whl</a></td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
    </tr>
    <tr>
        <td> win-cuda11.2-cudnn8.2-mkl-vs2019-avx </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.6.0/windows/windows-gpu-cuda11.2-cudnn8.2.1-mkl-avx-vs2019/paddlepaddle_gpu-2.6.0.post112-cp38-cp38-win_amd64.whl"> paddlepaddle_gpu-2.6.0.post112-cp38-cp38-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.6.0/windows/windows-gpu-cuda11.2-cudnn8.2.1-mkl-avx-vs2019/paddlepaddle_gpu-2.6.0.post112-cp39-cp39-win_amd64.whl"> paddlepaddle_gpu-2.6.0.post112-cp39-cp39-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.6.0/windows/windows-gpu-cuda11.2-cudnn8.2.1-mkl-avx-vs2019/paddlepaddle_gpu-2.6.0.post112-cp310-cp310-win_amd64.whl"> paddlepaddle_gpu-2.6.0.post112-cp310-cp310-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.6.0/windows/windows-gpu-cuda11.2-cudnn8.2.1-mkl-avx-vs2019/paddlepaddle_gpu-2.6.0.post112-cp311-cp311-win_amd64.whl"> paddlepaddle_gpu-2.6.0.post112-cp311-cp311-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.6.0/windows/windows-gpu-cuda11.2-cudnn8.2.1-mkl-avx-vs2019/paddlepaddle_gpu-2.6.0.post112-cp312-cp312-win_amd64.whl"> paddlepaddle_gpu-2.6.0.post112-cp312-cp312-win_amd64.whl</a></td>
    </tr>
    <tr>
        <td> win-cuda11.6-cudnn8.4-mkl-vs2019-avx </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.6.0/windows/windows-gpu-cuda11.6-cudnn8.4.0-mkl-avx-vs2019/paddlepaddle_gpu-2.6.0.post116-cp38-cp38-win_amd64.whl"> paddlepaddle_gpu-2.6.0.post116-cp38-cp38-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.6.0/windows/windows-gpu-cuda11.6-cudnn8.4.0-mkl-avx-vs2019/paddlepaddle_gpu-2.6.0.post116-cp39-cp39-win_amd64.whl"> paddlepaddle_gpu-2.6.0.post116-cp39-cp39-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.6.0/windows/windows-gpu-cuda11.6-cudnn8.4.0-mkl-avx-vs2019/paddlepaddle_gpu-2.6.0.post116-cp310-cp310-win_amd64.whl"> paddlepaddle_gpu-2.6.0.post116-cp310-cp310-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.6.0/windows/windows-gpu-cuda11.6-cudnn8.4.0-mkl-avx-vs2019/paddlepaddle_gpu-2.6.0.post116-cp311-cp311-win_amd64.whl"> paddlepaddle_gpu-2.6.0.post116-cp311-cp311-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.6.0/windows/windows-gpu-cuda11.6-cudnn8.4.0-mkl-avx-vs2019/paddlepaddle_gpu-2.6.0.post116-cp312-cp312-win_amd64.whl"> paddlepaddle_gpu-2.6.0.post116-cp312-cp312-win_amd64.whl</a></td>
    </tr>
    <tr>
        <td> win-cuda11.7-cudnn8.4-mkl-vs2019-avx </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.6.0/windows/windows-gpu-cuda11.7-cudnn8.4.1-mkl-avx-vs2019/paddlepaddle_gpu-2.6.0.post117-cp38-cp38-win_amd64.whl"> paddlepaddle_gpu-2.6.0.post117-cp38-cp38-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.6.0/windows/windows-gpu-cuda11.7-cudnn8.4.1-mkl-avx-vs2019/paddlepaddle_gpu-2.6.0.post117-cp39-cp39-win_amd64.whl"> paddlepaddle_gpu-2.6.0.post117-cp39-cp39-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.6.0/windows/windows-gpu-cuda11.7-cudnn8.4.1-mkl-avx-vs2019/paddlepaddle_gpu-2.6.0.post117-cp310-cp310-win_amd64.whl"> paddlepaddle_gpu-2.6.0.post117-cp310-cp310-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.6.0/windows/windows-gpu-cuda11.7-cudnn8.4.1-mkl-avx-vs2019/paddlepaddle_gpu-2.6.0.post117-cp311-cp311-win_amd64.whl"> paddlepaddle_gpu-2.6.0.post117-cp311-cp311-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.6.0/windows/windows-gpu-cuda11.7-cudnn8.4.1-mkl-avx-vs2019/paddlepaddle_gpu-2.6.0.post117-cp312-cp312-win_amd64.whl"> paddlepaddle_gpu-2.6.0.post117-cp312-cp312-win_amd64.whl</a></td>
    </tr>
    <tr>
        <td> win-cuda11.8-cudnn8.6-mkl-vs2019-avx </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.6.0/windows/windows-gpu-cuda11.8-cudnn8.6.0-mkl-avx-vs2019/paddlepaddle_gpu-2.6.0-cp38-cp38-win_amd64.whl"> paddlepaddle_gpu-2.6.0-cp38-cp38-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.6.0/windows/windows-gpu-cuda11.8-cudnn8.6.0-mkl-avx-vs2019/paddlepaddle_gpu-2.6.0-cp39-cp39-win_amd64.whl"> paddlepaddle_gpu-2.6.0-cp39-cp39-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.6.0/windows/windows-gpu-cuda11.8-cudnn8.6.0-mkl-avx-vs2019/paddlepaddle_gpu-2.6.0-cp310-cp310-win_amd64.whl"> paddlepaddle_gpu-2.6.0-cp310-cp310-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.6.0/windows/windows-gpu-cuda11.8-cudnn8.6.0-mkl-avx-vs2019/paddlepaddle_gpu-2.6.0-cp311-cp311-win_amd64.whl"> paddlepaddle_gpu-2.6.0-cp311-cp311-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.6.0/windows/windows-gpu-cuda11.8-cudnn8.6.0-mkl-avx-vs2019/paddlepaddle_gpu-2.6.0-cp312-cp312-win_amd64.whl"> paddlepaddle_gpu-2.6.0-cp312-cp312-win_amd64.whl</a></td>
    </tr>
    <tr>
        <td> win-cuda12.0-cudnn8.9-mkl-vs2019-avx </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.6.0/windows/windows-gpu-cuda12.0-cudnn8.9.1-mkl-avx-vs2019/paddlepaddle_gpu-2.6.0.post120-cp38-cp38-win_amd64.whl"> paddlepaddle_gpu-2.6.0.post120-cp38-cp38-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.6.0/windows/windows-gpu-cuda12.0-cudnn8.9.1-mkl-avx-vs2019/paddlepaddle_gpu-2.6.0.post120-cp39-cp39-win_amd64.whl"> paddlepaddle_gpu-2.6.0.post120-cp39-cp39-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.6.0/windows/windows-gpu-cuda12.0-cudnn8.9.1-mkl-avx-vs2019/paddlepaddle_gpu-2.6.0.post120-cp310-cp310-win_amd64.whl"> paddlepaddle_gpu-2.6.0.post120-cp310-cp310-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.6.0/windows/windows-gpu-cuda12.0-cudnn8.9.1-mkl-avx-vs2019/paddlepaddle_gpu-2.6.0.post120-cp311-cp311-win_amd64.whl"> paddlepaddle_gpu-2.6.0.post120-cp311-cp311-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.6.0/windows/windows-gpu-cuda12.0-cudnn8.9.1-mkl-avx-vs2019/paddlepaddle_gpu-2.6.0.post120-cp312-cp312-win_amd64.whl"> paddlepaddle_gpu-2.6.0.post120-cp312-cp312-win_amd64.whl</a></td>
    </tr>
    <tr>
        <td> linux-cinn-cuda11.2-cudnn8-mkl-gcc8.2-avx </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.6.0/linux-cinn/linux-gpu-cuda11.2-cudnn8-mkl-gcc8.2-avx/paddlepaddle_gpu-2.6.0.post112-cp38-cp38-linux_x86_64.whl"> paddlepaddle_gpu-2.6.0.post112-cp38-cp38-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.6.0/linux-cinn/linux-gpu-cuda11.2-cudnn8-mkl-gcc8.2-avx/paddlepaddle_gpu-2.6.0.post112-cp39-cp39-linux_x86_64.whl"> paddlepaddle_gpu-2.6.0.post112-cp39-cp39-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.6.0/linux-cinn/linux-gpu-cuda11.2-cudnn8-mkl-gcc8.2-avx/paddlepaddle_gpu-2.6.0.post112-cp310-cp310-linux_x86_64.whl"> paddlepaddle_gpu-2.6.0.post112-cp310-cp310-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.6.0/linux-cinn/linux-gpu-cuda11.2-cudnn8-mkl-gcc8.2-avx/paddlepaddle_gpu-2.6.0.post112-cp311-cp311-linux_x86_64.whl"> paddlepaddle_gpu-2.6.0.post112-cp311-cp311-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.6.0/linux-cinn/linux-gpu-cuda11.2-cudnn8-mkl-gcc8.2-avx/paddlepaddle_gpu-2.6.0.post112-cp312-cp312-linux_x86_64.whl"> paddlepaddle_gpu-2.6.0.post112-cp312-cp312-linux_x86_64.whl</a></td>
        <td> - </td>
    </tr>
    <tr>
        <td> linux-cuda11.2-cudnn8-mkl-gcc8.2-avx-pascal </td>
        <td> - </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.6.0/linux-pascal/linux-gpu-cuda11.2-cudnn8-mkl-gcc8.2-avx/paddlepaddle_gpu-2.6.0.post112-cp38-cp38-linux_x86_64.whl"> paddlepaddle_gpu-2.6.0-cp38-cp38-linux_x86_64.whl</a> </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.6.0/linux-pascal/linux-gpu-cuda11.2-cudnn8-mkl-gcc8.2-avx/paddlepaddle_gpu-2.6.0.post112-cp39-cp39-linux_x86_64.whl"> paddlepaddle_gpu-2.6.0-cp39-cp39-linux_x86_64.whl</a> </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.6.0/linux-pascal/linux-gpu-cuda11.2-cudnn8-mkl-gcc8.2-avx/paddlepaddle_gpu-2.6.0.post112-cp310-cp310-linux_x86_64.whl"> paddlepaddle_gpu-2.6.0-cp310-cp310-linux_x86_64.whl</a> </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.6.0/linux-pascal/linux-gpu-cuda11.2-cudnn8-mkl-gcc8.2-avx/paddlepaddle_gpu-2.6.0.post112-cp311-cp311-linux_x86_64.whl"> paddlepaddle_gpu-2.6.0-cp311-cp311-linux_x86_64.whl</a> </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.6.0/linux-pascal/linux-gpu-cuda11.2-cudnn8-mkl-gcc8.2-avx/paddlepaddle_gpu-2.6.0.post112-cp312-cp312-linux_x86_64.whl"> paddlepaddle_gpu-2.6.0-cp312-cp312-linux_x86_64.whl</a> </td>
    </tr>
   </tbody>
</table>
</p>

### 表格说明

- 纵轴

cpu-mkl: 支持 CPU 训练和预测，使用 Intel mkl 数学库

cuda10_cudnn7-mkl: 支持 GPU 训练和预测，使用 Intel mkl 数学库


- 横轴

一般是类似于“cp310-cp310”的形式，其中：

310:python tag,指 python3.10，类似的还有“38”、“39”、“311”、“312”等

mu:指 unicode 版本 python，若为 m 则指非 unicode 版本 python

- 安装包命名规则

每个安装包都有一个专属的名字，它们是按照 Python 的官方规则 来命名的，形式如下：

{distribution}-{version}(-{build tag})?-{python tag}-{abi tag}-{platform tag}.whl

其中 build tag 可以缺少，其他部分不能缺少

distribution: wheel 名称 version: 版本，例如 0.14.0 (要求必须是数字格式)

python tag: 类似'py38', 'py38', 'py310', 'py311', 'py312'，'py310'，用于标明对应的 python 版本

abi tag:  类似'cp33m', 'abi3', 'none'

platform tag: 类似 'linux_x86_64', 'any'

<a name="ciwhls"></a>
</br></br>
## **多版本 whl 包列表-develop**
<p align="center">
<table>
    <thead>
    <tr>
        <th> 版本说明 </th>
        <th> cp38-cp38    </th>
        <th> cp39-cp39    </th>
        <th> cp310-cp310    </th>
        <th> cp311-cp311    </th>
        <th> cp312-cp312    </th>
    </tr>
    </thead>
    <tbody>
    <tr>
        <td> linux-cpu-mkl-avx </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/linux/linux-cpu-mkl-avx/paddlepaddle-0.0.0-cp38-cp38-linux_x86_64.whl"> paddlepaddle-latest-cp38-cp38-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/linux/linux-cpu-mkl-avx/paddlepaddle-0.0.0-cp39-cp39-linux_x86_64.whl"> paddlepaddle-latest-cp39-cp39-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/linux/linux-cpu-mkl-avx/paddlepaddle-0.0.0-cp310-cp310-linux_x86_64.whl"> paddlepaddle-latest-cp310-cp310-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/linux/linux-cpu-mkl-avx/paddlepaddle-0.0.0-cp311-cp311-linux_x86_64.whl"> paddlepaddle-latest-cp311-cp311-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/linux/linux-cpu-mkl-avx/paddlepaddle-0.0.0-cp312-cp312-linux_x86_64.whl"> paddlepaddle-latest-cp312-cp312-linux_x86_64.whl</a></td>
    </tr>
    <tr>
        <td> linux-cpu-openblas-avx </td>
        <td> - </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/linux/linux-cpu-openblas-avx/paddlepaddle-0.0.0-cp38-cp38-linux_x86_64.whl"> paddlepaddle-latest-cp38-cp38-linux_x86_64.whl</a></td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
    </tr>
    <tr>
        <td> cuda11.2-cudnn8.1-mkl </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/linux/linux-gpu-cuda11.2-cudnn8-mkl-gcc8.2-avx/paddlepaddle_gpu-0.0.0.post112-cp38-cp38-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp38-cp38-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/linux/linux-gpu-cuda11.2-cudnn8-mkl-gcc8.2-avx/paddlepaddle_gpu-0.0.0.post112-cp39-cp39-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp39-cp39-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/linux/linux-gpu-cuda11.2-cudnn8-mkl-gcc8.2-avx/paddlepaddle_gpu-0.0.0.post112-cp310-cp310-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp310-cp310-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/linux/linux-gpu-cuda11.2-cudnn8-mkl-gcc8.2-avx/paddlepaddle_gpu-0.0.0.post112-cp311-cp311-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp311-cp311-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/linux/linux-gpu-cuda11.2-cudnn8-mkl-gcc8.2-avx/paddlepaddle_gpu-0.0.0.post112-cp312-cp312-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp312-cp312-linux_x86_64.whl</a></td>
    </tr>
    <tr>
        <td> cuda11.6-cudnn8.4-mkl </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/linux/linux-gpu-cuda11.6-cudnn8.4.0-mkl-gcc8.2-avx/paddlepaddle_gpu-0.0.0.post116-cp38-cp38-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp38-cp38-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/linux/linux-gpu-cuda11.6-cudnn8.4.0-mkl-gcc8.2-avx/paddlepaddle_gpu-0.0.0.post116-cp39-cp39-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp39-cp39-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/linux/linux-gpu-cuda11.6-cudnn8.4.0-mkl-gcc8.2-avx/paddlepaddle_gpu-0.0.0.post116-cp310-cp310-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp310-cp310-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/linux/linux-gpu-cuda11.6-cudnn8.4.0-mkl-gcc8.2-avx/paddlepaddle_gpu-0.0.0.post116-cp311-cp311-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp311-cp311-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/linux/linux-gpu-cuda11.6-cudnn8.4.0-mkl-gcc8.2-avx/paddlepaddle_gpu-0.0.0.post116-cp312-cp312-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp312-cp312-linux_x86_64.whl</a></td>
    </tr>
    <tr>
        <td> cuda11.7-cudnn8.4-mkl </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/linux/linux-gpu-cuda11.7-cudnn8.4.1-mkl-gcc8.2-avx/paddlepaddle_gpu-0.0.0.post117-cp38-cp38-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp38-cp38-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/linux/linux-gpu-cuda11.7-cudnn8.4.1-mkl-gcc8.2-avx/paddlepaddle_gpu-0.0.0.post117-cp39-cp39-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp39-cp39-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/linux/linux-gpu-cuda11.7-cudnn8.4.1-mkl-gcc8.2-avx/paddlepaddle_gpu-0.0.0.post117-cp310-cp310-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp310-cp310-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/linux/linux-gpu-cuda11.7-cudnn8.4.1-mkl-gcc8.2-avx/paddlepaddle_gpu-0.0.0.post117-cp311-cp311-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp311-cp311-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/linux/linux-gpu-cuda11.7-cudnn8.4.1-mkl-gcc8.2-avx/paddlepaddle_gpu-0.0.0.post117-cp312-cp312-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp312-cp312-linux_x86_64.whl</a></td>
    </tr>
    <tr>
        <td> cuda11.8-cudnn8.6-mkl </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/linux/linux-gpu-cuda11.8-cudnn8.6-mkl-gcc8.2-avx/paddlepaddle_gpu-0.0.0.post118-cp38-cp38-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp38-cp38-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/linux/linux-gpu-cuda11.8-cudnn8.6-mkl-gcc8.2-avx/paddlepaddle_gpu-0.0.0.post118-cp39-cp39-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp39-cp39-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/linux/linux-gpu-cuda11.8-cudnn8.6-mkl-gcc8.2-avx/paddlepaddle_gpu-0.0.0.post118-cp310-cp310-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp310-cp310-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/linux/linux-gpu-cuda11.8-cudnn8.6-mkl-gcc8.2-avx/paddlepaddle_gpu-0.0.0.post118-cp311-cp311-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp311-cp311-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/linux/linux-gpu-cuda11.8-cudnn8.6-mkl-gcc8.2-avx/paddlepaddle_gpu-0.0.0.post118-cp312-cp312-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp312-cp312-linux_x86_64.whl</a></td>
    </tr>
    <tr>
        <td> cuda12.0-cudnn8.9-mkl </td>
        <td> - </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/linux/linux-gpu-cuda12.0-cudnn8.9-mkl-gcc12.2-avx/paddlepaddle_gpu-0.0.0.post120-cp38-cp38-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp38-cp38-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/linux/linux-gpu-cuda12.0-cudnn8.9-mkl-gcc12.2-avx/paddlepaddle_gpu-0.0.0.post120-cp39-cp39-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp39-cp39-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/linux/linux-gpu-cuda12.0-cudnn8.9-mkl-gcc12.2-avx/paddlepaddle_gpu-0.0.0.post120-cp310-cp310-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp310-cp310-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/linux/linux-gpu-cuda12.0-cudnn8.9-mkl-gcc12.2-avx/paddlepaddle_gpu-0.0.0.post120-cp311-cp311-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp311-cp311-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/linux/linux-gpu-cuda12.0-cudnn8.9-mkl-gcc12.2-avx/paddlepaddle_gpu-0.0.0.post120-cp312-cp312-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp312-cp312-linux_x86_64.whl</a></td>
    </tr>
    <tr>
        <td> mac-cpu </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/macos/macos-cpu-openblas/paddlepaddle-0.0.0-cp38-cp38-macosx_10_9_x86_64.whl"> paddlepaddle-cp38-cp38-macosx_10_9_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/macos/macos-cpu-openblas/paddlepaddle-0.0.0-cp39-cp39-macosx_10_9_x86_64.whl"> paddlepaddle-cp39-cp39-macosx_10_9_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/macos/macos-cpu-openblas/paddlepaddle-0.0.0-cp310-cp310-macosx_10_9_x86_64.whl"> paddlepaddle-cp310-cp310-macosx_10_9_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/macos/macos-cpu-openblas/paddlepaddle-0.0.0-cp311-cp311-macosx_10_9_x86_64.whl"> paddlepaddle-cp311-cp311-macosx_10_9_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/macos/macos-cpu-openblas/paddlepaddle-0.0.0-cp312-cp312-macosx_10_9_x86_64.whl"> paddlepaddle-cp312-cp312-macosx_10_9_x86_64.whl</a></td>
    </tr>
    <tr>
        <td> macos-cpu-openblas-m1 </td>
        <td> - </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/macos/macos-cpu-openblas-m1/paddlepaddle-0.0.0-cp38-cp38-macosx_11_0_arm64.whl"> paddlepaddle-cp38-cp38-macosx_11_0_arm64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/macos/macos-cpu-openblas-m1/paddlepaddle-0.0.0-cp39-cp39-macosx_11_0_arm64.whl"> paddlepaddle-cp39-cp39-macosx_11_0_arm64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/macos/macos-cpu-openblas-m1/paddlepaddle-0.0.0-cp310-cp310-macosx_11_0_arm64.whl"> paddlepaddle-cp310-cp310-macosx_11_0_arm64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/macos/macos-cpu-openblas-m1/paddlepaddle-0.0.0-cp311-cp311-macosx_11_0_arm64.whl"> paddlepaddle-cp311-cp311-macosx_11_0_arm64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/macos/macos-cpu-openblas-m1/paddlepaddle-0.0.0-cp312-cp312-macosx_11_0_arm64.whl"> paddlepaddle-cp312-cp312-macosx_11_0_arm64.whl</a></td>
    </tr>
    <tr>
        <td> win-cpu-mkl-avx </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/windows/windows-cpu-avx-mkl-vs2017/paddlepaddle-0.0.0-cp38-cp38-win_amd64.whl"> paddlepaddle-latest-cp38-cp38-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/windows/windows-cpu-avx-mkl-vs2017/paddlepaddle-0.0.0-cp39-cp39-win_amd64.whl"> paddlepaddle-latest-cp39-cp39-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/windows/windows-cpu-avx-mkl-vs2017/paddlepaddle-0.0.0-cp310-cp310-win_amd64.whl"> paddlepaddle-latest-cp310-cp310-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/windows/windows-cpu-avx-mkl-vs2017/paddlepaddle-0.0.0-cp311-cp311-win_amd64.whl"> paddlepaddle-latest-cp311-cp311-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/windows/windows-cpu-avx-mkl-vs2017/paddlepaddle-0.0.0-cp312-cp312-win_amd64.whl"> paddlepaddle-latest-cp312-cp312-win_amd64.whl</a></td>
    </tr>
    <tr>
        <td> win-cpu-openblas-avx </td>
        <td> - </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/windows/cpu-openblas-avx/paddlepaddle-0.0.0-cp38-cp38-win_amd64.whl"> paddlepaddle-latest-cp38-cp38-win_amd64.whl</a></td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
    </tr>
    <tr>
    <tr>
        <td> win-cuda11.2-cudnn8.2-mkl-vs2019-avx </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/windows/windows-gpu-cuda11.2-cudnn8.2.1-mkl-avx-vs2019/paddlepaddle_gpu-0.0.0.post112-cp38-cp38-win_amd64.whl"> paddlepaddle_gpu-latest-cp38-cp38-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/windows/windows-gpu-cuda11.2-cudnn8.2.1-mkl-avx-vs2019/paddlepaddle_gpu-0.0.0.post112-cp39-cp39-win_amd64.whl"> paddlepaddle_gpu-latest-cp39-cp39-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/windows/windows-gpu-cuda11.2-cudnn8.2.1-mkl-avx-vs2019/paddlepaddle_gpu-0.0.0.post112-cp310-cp310-win_amd64.whl"> paddlepaddle_gpu-latest-cp310-cp310-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/windows/windows-gpu-cuda11.2-cudnn8.2.1-mkl-avx-vs2019/paddlepaddle_gpu-0.0.0.post112-cp311-cp311-win_amd64.whl"> paddlepaddle_gpu-latest-cp311-cp311-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/windows/windows-gpu-cuda11.2-cudnn8.2.1-mkl-avx-vs2019/paddlepaddle_gpu-0.0.0.post112-cp312-cp312-win_amd64.whl"> paddlepaddle_gpu-latest-cp312-cp312-win_amd64.whl</a></td>
    </tr>
    <tr>
        <td> win-cuda11.6-cudnn8.4.0-mkl-avx-vs2019 </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/windows/windows-gpu-cuda11.6-cudnn8.4.0-mkl-avx-vs2019/paddlepaddle_gpu-0.0.0.post116-cp38-cp38-win_amd64.whl"> paddlepaddle_gpu-latest-cp38-cp38-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/windows/windows-gpu-cuda11.6-cudnn8.4.0-mkl-avx-vs2019/paddlepaddle_gpu-0.0.0.post116-cp39-cp39-win_amd64.whl"> paddlepaddle_gpu-latest-cp39-cp39-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/windows/windows-gpu-cuda11.6-cudnn8.4.0-mkl-avx-vs2019/paddlepaddle_gpu-0.0.0.post116-cp310-cp310-win_amd64.whl"> paddlepaddle_gpu-latest-cp310-cp310-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/windows/windows-gpu-cuda11.6-cudnn8.4.0-mkl-avx-vs2019/paddlepaddle_gpu-0.0.0.post116-cp311-cp311-win_amd64.whl"> paddlepaddle_gpu-latest-cp311-cp311-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/windows/windows-gpu-cuda11.6-cudnn8.4.0-mkl-avx-vs2019/paddlepaddle_gpu-0.0.0.post116-cp312-cp312-win_amd64.whl"> paddlepaddle_gpu-latest-cp312-cp312-win_amd64.whl</a></td>
    </tr>
    <tr>
        <td> win-cuda11.7-cudnn8.4.1-mkl-avx-vs2019 </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/windows/windows-gpu-cuda11.7-cudnn8.4.1-mkl-avx-vs2019/paddlepaddle_gpu-0.0.0.post117-cp38-cp38-win_amd64.whl"> paddlepaddle_gpu-latest-cp38-cp38-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/windows/windows-gpu-cuda11.7-cudnn8.4.1-mkl-avx-vs2019/paddlepaddle_gpu-0.0.0.post117-cp39-cp39-win_amd64.whl"> paddlepaddle_gpu-latest-cp39-cp39-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/windows/windows-gpu-cuda11.7-cudnn8.4.1-mkl-avx-vs2019/paddlepaddle_gpu-0.0.0.post117-cp310-cp310-win_amd64.whl"> paddlepaddle_gpu-latest-cp310-cp310-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/windows/windows-gpu-cuda11.7-cudnn8.4.1-mkl-avx-vs2019/paddlepaddle_gpu-0.0.0.post117-cp311-cp311-win_amd64.whl"> paddlepaddle_gpu-latest-cp311-cp311-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/windows/windows-gpu-cuda11.7-cudnn8.4.1-mkl-avx-vs2019/paddlepaddle_gpu-0.0.0.post117-cp312-cp312-win_amd64.whl"> paddlepaddle_gpu-latest-cp312-cp312-win_amd64.whl</a></td>
    </tr>
    <tr>
        <td> win-cuda11.8-cudnn8.6.0-mkl-avx-vs2019 </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/windows/windows-gpu-cuda11.8-cudnn8.6.0-mkl-avx-vs2019/paddlepaddle_gpu-0.0.0.post118-cp38-cp38-win_amd64.whl"> paddlepaddle_gpu-latest-cp38-cp38-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/windows/windows-gpu-cuda11.8-cudnn8.6.0-mkl-avx-vs2019/paddlepaddle_gpu-0.0.0.post118-cp39-cp39-win_amd64.whl"> paddlepaddle_gpu-latest-cp39-cp39-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/windows/windows-gpu-cuda11.8-cudnn8.6.0-mkl-avx-vs2019/paddlepaddle_gpu-0.0.0.post118-cp310-cp310-win_amd64.whl"> paddlepaddle_gpu-latest-cp310-cp310-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/windows/windows-gpu-cuda11.8-cudnn8.6.0-mkl-avx-vs2019/paddlepaddle_gpu-0.0.0.post118-cp311-cp311-win_amd64.whl"> paddlepaddle_gpu-latest-cp311-cp311-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/windows/windows-gpu-cuda11.8-cudnn8.6.0-mkl-avx-vs2019/paddlepaddle_gpu-0.0.0.post118-cp312-cp312-win_amd64.whl"> paddlepaddle_gpu-latest-cp312-cp312-win_amd64.whl</a></td>
    </tr>
        <tr>
        <td> win-cuda12.0-cudnn8.9.1-mkl-avx-vs2019 </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/windows/windows-gpu-cuda12.0-cudnn8.9.1-mkl-avx-vs2019/paddlepaddle_gpu-0.0.0.post120-cp38-cp38-win_amd64.whl"> paddlepaddle_gpu-latest-cp38-cp38-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/windows/windows-gpu-cuda12.0-cudnn8.9.1-mkl-avx-vs2019/paddlepaddle_gpu-0.0.0.post120-cp39-cp39-win_amd64.whl"> paddlepaddle_gpu-latest-cp39-cp39-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/windows/windows-gpu-cuda12.0-cudnn8.9.1-mkl-avx-vs2019/paddlepaddle_gpu-0.0.0.post120-cp310-cp310-win_amd64.whl"> paddlepaddle_gpu-latest-cp310-cp310-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/windows/windows-gpu-cuda12.0-cudnn8.9.1-mkl-avx-vs2019/paddlepaddle_gpu-0.0.0.post120-cp311-cp311-win_amd64.whl"> paddlepaddle_gpu-latest-cp311-cp311-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/windows/windows-gpu-cuda12.0-cudnn8.9.1-mkl-avx-vs2019/paddlepaddle_gpu-0.0.0.post120-cp312-cp312-win_amd64.whl"> paddlepaddle_gpu-latest-cp312-cp312-win_amd64.whl</a></td>
    </tr>
   </tbody>
</table>
</p>




<!--TODO this part should be in a new webpage-->

</br></br>

## 在 Docker 中执行 PaddlePaddle 训练程序


假设您已经在当前目录（比如在/home/work）编写了一个 PaddlePaddle 的程序: `train.py` （可以参考
[PaddlePaddleBook](https://github.com/PaddlePaddle/book/blob/develop/01.fit_a_line/README.cn.md)
编写），就可以使用下面的命令开始执行训练：

```
cd /home/work
```
```
docker run -it -v $PWD:/work registry.baidubce.com/paddlepaddle/paddle /work/train.py
```

上述命令中，`-it` 参数说明容器已交互式运行；`-v $PWD:/work`
指定将当前路径（Linux 中 PWD 变量会展开为当前路径的绝对路径）挂载到容器内部的:`/work`
目录: `registry.baidubce.com/paddlepaddle/paddle` 指定需要使用的容器； 最后`/work/train.py`为容器内执行的命令，即运行训练程序。

当然，您也可以进入到 Docker 容器中，以交互式的方式执行或调试您的代码：

```
docker run -it -v $PWD:/work registry.baidubce.com/paddlepaddle/paddle /bin/bash
```
```
cd /work
```
```
python train.py
```

**注：PaddlePaddle Docker 镜像为了减小体积，默认没有安装 vim，您可以在容器中执行** `apt-get install -y vim` **安装后，在容器中编辑代码。**

</br></br>

## 使用 Docker 启动 PaddlePaddle Book 教程


使用 Docker 可以快速在本地启动一个包含了 PaddlePaddle 官方 Book 教程的 Jupyter Notebook，可以通过网页浏览。
PaddlePaddle Book 是为用户和开发者制作的一个交互式的 Jupyter Notebook。
如果您想要更深入了解 deep learning，可以参考 PaddlePaddle Book。
大家可以通过它阅读教程，或者制作和分享带有代码、公式、图表、文字的交互式文档。

我们提供可以直接运行 PaddlePaddle Book 的 Docker 镜像，直接运行：

```
docker run -p 8888:8888 registry.baidubce.com/paddlepaddle/book
```

国内用户可以使用下面的镜像源来加速访问：

```
docker run -p 8888:8888 registry.baidubce.com/paddlepaddle/book
```

然后在浏览器中输入以下网址：

```
http://localhost:8888/
```


</br></br>
## 使用 Docker 执行 GPU 训练


为了保证 GPU 驱动能够在镜像里面正常运行，我们推荐使用
[nvidia-docker](https://github.com/NVIDIA/nvidia-docker)来运行镜像。
请不要忘记提前在物理机上安装 GPU 最新驱动。

```
nvidia-docker run -it -v $PWD:/work registry.baidubce.com/paddlepaddle/paddle:latest-gpu /bin/bash
```

**注: 如果没有安装 nvidia-docker，可以尝试以下的方法，将 CUDA 库和 Linux 设备挂载到 Docker 容器内：**

```
export CUDA_SO="$(\ls /usr/lib64/libcuda* | xargs -I{} echo '-v {}:{}') \
$(\ls /usr/lib64/libnvidia* | xargs -I{} echo '-v {}:{}')"
export DEVICES=$(\ls /dev/nvidia* | xargs -I{} echo '--device {}:{}')
docker run ${CUDA_SO} \
${DEVICES} -it registry.baidubce.com/paddlepaddle/paddle:latest-gpu
```
