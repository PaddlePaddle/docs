<a name="third_party"></a>
# Appendix


## Compile Dependency Table

<p align="center">
<table>
    <thead>
    <tr>
        <th> Dependency package name </th>
        <th> Version </th>
        <th> Description </th>
        <th> Installation command </th>
    </tr>
    </thead>
    <tbody>
    <tr>
        <td> CMake </td>
        <td> 3.10, 3.11, 3.15, 3.16(Recommend),3.17 </td>
        <td> There is an official bug in version 3.12/3.13/3.14, please skip this version </td>
        <td>  </td>
    </tr>
    <tr>
        <td> GCC </td>
        <td> 4.8 / 5.4 </td>
        <td>  recommends using devtools2 for CentOS </td>
        <td>  </td>
    </tr>
    <tr>
        <td> Clang (MacOS Only) </td>
        <td> 9.0 and above </td>
        <td> Usually use the clang version of MacOS 10.11 and above </td>
        <td>  </td>
    </tr>
        <tr>
        <td> Python（64 bit） </td>
        <td> 2.7.x. or 3.5+.x </td>
        <td> depends on libpython2.7.so or libpython3.5+.so </td>
        <td> <code> apt install python-dev </code> or <code> yum install python-devel </code> if installing python3, please go to <a href="https://www.python.org">Python official website </a></td>
    </tr>
    <tr>
        <td> SWIG </td>
        <td> at least 2.0 </td>
        <td>  </td>
        <td> <code>apt install swig </code> or <code> yum install swig </code> </td>
    </tr>
    <tr>
        <td> wget </td>
        <td> any </td>
        <td>  </td>
        <td> <code> apt install wget </code>  or <code> yum install wget </code> </td>
    </tr>
    <tr>
        <td> openblas </td>
        <td> any </td>
        <td> optional </td>
        <td>  </td>
    </tr>
    <tr>
        <td> pip </td>
        <td> at least 9.0.1 </td>
        <td>  </td>
        <td> <code> apt install python-pip </code> or <code> yum install Python-pip </code> </td>
    </tr>
    <tr>
        <td> numpy </td>
        <td> >=1.12.0 </td>
        <td>  </td>
        <td> <code> pip install numpy </code> </td>
    </tr>
    <tr>
        <td> protobuf </td>
        <td> >=3.1.0 </td>
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
        <td> <code> apt install patchelf </code> or read github <a href="https://gist.github.com/ruario/80fefd174b3395d34c14">patchELF official documentation</a></td>
    </tr>
    <tr>
        <td> go </td>
        <td> >=1.8 </td>
        <td> optional </td>
        <td>  </td>
    </tr>
    <tr>
        <td> setuptools </td>
        <td> >= 28.0.0 </td>
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
## **Compile Option Table**

<p align="center">
<table>
    <thead>
    <tr>
        <th> Option </th>
        <th> Description  </th>
        <th> Default </th>
    </tr>
    </thead>
    <tbody>
    <tr>
        <td> WITH_GPU </td>
        <td> Whether to support GPU </td>
        <td> ON </td>
    </tr>
    <tr>
        <td> WITH_AVX </td>
        <td> whether to compile PaddlePaddle binaries file containing the AVX instruction set </td>
        <td> ON </td>
    </tr>
    <tr>
        <td> WITH_PYTHON </td>
        <td> Whether the PYTHON interpreter is embedded </td>
        <td> ON </td>
    </tr>
    <tr>
        <td> WITH_TESTING </td>
        <td>  Whether to turn on unit test </td>
        <td> OFF </td>
    </tr>
    <tr>
        <td> WITH_MKL </td>
        <td> Whether to use the MKL math library, if not,using OpenBLAS </td>
        <td> ON </td>
    </tr>
    <tr>
        <td> WITH_SYSTEM_BLAS </td>
        <td> Whether to use the system's BLAS </td>
        <td> OFF </td>
    </tr>
    <tr>
        <td> WITH_DISTRIBUTE </td>
        <td> Whether to Compile with distributed version </td>
        <td> OFF </td>
    </tr>
    <tr>
        <td> WITH_BRPC_RDMA </td>
        <td> Whether to use BRPC RDMA as RPC protocol </td>
        <td> OFF </td>
    </tr>
        <tr>
        <td> ON_INFER </td>
        <td> Whether to turn on prediction optimization </td>
        <td> OFF </td>
    </tr>
    <tr>
        <tr>
        <td> CUDA_ARCH_NAME </td>
        <td> Compile only for current CUDA schema or not</td>
        <td> All:Compile all supported CUDA architectures  optional: Auto automatically recognizes the schema compilation of the current environment</td>
    </tr>
    <tr>
        <tr>
        <td> TENSORRT_ROOT </td>
        <td> Specify TensorRT path </td>
        <td> The default value under windows is '/', The default value under windows is '/usr/' </td>
    </tr>



**BLAS**

PaddlePaddle supports two BLAS libraries, [MKL](https://software.intel.com/en-us/mkl) and [OpenBlAS](http://www.openblas.net/). MKL is used by default. If you use MKL and the machine contains the AVX2 instruction set, you will also download the MKL-DNN math library, for details please refer to [here](https://github.com/PaddlePaddle/Paddle/tree/release/0.11.0/doc/design/mkldnn#cmake).

If you close MKL, OpenBLAS will be used as the BLAS library.

**CUDA/cuDNN**

PaddlePaddle automatically finds the CUDA and cuDNN libraries installed in the system for compilation and execution at compile time/runtime. Use the parameter `-DCUDA_ARCH_NAME=Auto` to specify to enable automatic detection of the SM architecture and speed up compilation.

PaddlePaddle can be compiled and run using any version after cuDNN v5.1, but try to keep the same version of cuDNN in the compiling and running processes. We recommend using the latest version of cuDNN.

**Configure Compile Options**

PaddePaddle implements references to various BLAS/CUDA/cuDNN libraries by specifying paths at compile time. When cmake compiles, it first searches the system paths ( `/usr/liby` and `/usr/local/lib` ) for these libraries, and also reads the relevant path variables for searching. Can be set by using the `-D` command, for example:

> `Cmake .. -DWITH_GPU=ON -DWITH_TESTING=OFF -DCUDNN_ROOT=/opt/cudnnv5`

**Note**: The settings introduced here for these compilation options are only valid for the first cmake. If you want to reset it later, it is recommended to clean up the entire build directory ( rm -rf ) and then specify it.


<a name="whls"></a>
</br></br>
## **Installation Package List**


<p align="center">
<table>
    <thead>
    <tr>
        <th> Version Number </th>
        <th> Release Discription </th>
    </tr>
    </thead>
    <tbody>
    <tr>
        <td> paddlepaddle==[version code] such as paddlepaddle==2.0.0a0 </td>
        <td> Only support the corresponding version of the CPU PaddlePaddle, please refer to <a href=https://pypi.org/project/paddlepaddle/#history>Pypi</a> for the specific version. </td>
    </tr>
    <tr>
        <td> paddlepaddle-gpu==[version code], such as paddlepaddle-gpu==2.0.0a0 </td>
        <td> The default installation supports the PaddlePaddle installation package corresponding to [version number] of CUDA 10.0 and cuDNN 7 </td>
    </tr>
   </tbody>
</table>
</p>

You can find various distributions of PaddlePaddle-gpu in [the Release History](https://pypi.org/project/paddlepaddle-gpu/#history).
> 'postxx' corresponds to CUDA and cuDNN versions, and the number before 'postxx' represents the version of Paddle

Please note that: in the commands, <code> paddlepaddle-gpu==2.0.0a0 </code> will install the installation package of PaddlePaddle that supports CUDA 10.0 and cuDNN 7 by default under Windows environment.


<a name="ciwhls-release"></a>
</br></br>

## **Multi-version whl package list - Release**


<p align="center">
<table>
    <thead>
    <tr>
        <th> Release Instruction </th>
        <th> cp27-cp27mu </th>
        <th> cp27-cp27m </th>
        <th> cp35-cp35m    </th>
        <th> cp36-cp36m    </th>
        <th> cp37-cp37m    </th>
    </tr>
    </thead>
    <tbody>
    <tr>
        <td> cpu-mkl </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.0.0-alpha0-cpu-mkl/paddlepaddle-2.0.0a0-cp27-cp27mu-linux_x86_64.whl">
        paddlepaddle-2.0.0a0-cp27-cp27mu-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.0.0-alpha0-cpu-mkl/paddlepaddle-2.0.0a0-cp27-cp27m-linux_x86_64.whl">
        paddlepaddle-2.0.0a0-cp27-cp27m-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.0.0-alpha0-cpu-mkl/paddlepaddle-2.0.0a0-cp35-cp35m-linux_x86_64.whl">
        paddlepaddle-2.0.0a0-cp35-cp35m-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.0.0-alpha0-cpu-mkl/paddlepaddle-2.0.0a0-cp36-cp36m-linux_x86_64.whl">
        paddlepaddle-2.0.0a0-cp36-cp36m-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.0.0-alpha0-cpu-mkl/paddlepaddle-2.0.0a0-cp37-cp37m-linux_x86_64.whl">
        paddlepaddle-2.0.0a0-cp37-cp37m-linux_x86_64.whl</a></td>
    </tr>
    <tr>
        <td> cuda10_cudnn7-mkl </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.0.0-alpha0-gpu-cuda10-cudnn7-mkl/paddlepaddle_gpu-2.0.0a0-cp27-cp27mu-linux_x86_64.whl">
        paddlepaddle_gpu-2.0.0a0-cp27-cp27mu-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.0.0-alpha0-gpu-cuda10-cudnn7-mkl/paddlepaddle_gpu-2.0.0a0-cp27-cp27m-linux_x86_64.whl">
        paddlepaddle_gpu-2.0.0a0-cp27-cp27m-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.0.0-alpha0-gpu-cuda10-cudnn7-mkl/paddlepaddle_gpu-2.0.0a0-cp35-cp35m-linux_x86_64.whl">
        paddlepaddle_gpu-2.0.0a0-cp35-cp35m-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.0.0-alpha0-gpu-cuda10-cudnn7-mkl/paddlepaddle_gpu-2.0.0a0-cp36-cp36m-linux_x86_64.whl">
        paddlepaddle_gpu-2.0.0a0-cp36-cp36m-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.0.0-alpha0-gpu-cuda10-cudnn7-mkl/paddlepaddle_gpu-2.0.0a0-cp37-cp37m-linux_x86_64.whl">
        paddlepaddle_gpu-2.0.0a0-cp37-cp37m-linux_x86_64.whl</a></td>
    </tr>
    <tr>
        <td> win_cpu_mkl </td>
        <td> - </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.0.0/win-mkl/paddlepaddle-2.0.0a0-cp27-cp27m-win_amd64.whl">
        paddlepaddle-2.0.0a0-cp27-cp27m-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.0.0/win-mkl/paddlepaddle-2.0.0a0-cp35-cp35m-win_amd64.whl">
        paddlepaddle-2.0.0a0-cp35-cp35m-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.0.0/win-mkl/paddlepaddle-2.0.0a0-cp36-cp36m-win_amd64.whl">
        paddlepaddle-2.0.0a0-cp36-cp36m-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.0.0/win-mkl/paddlepaddle-2.0.0a0-cp37-cp37m-win_amd64.whl">
        paddlepaddle-2.0.0a0-cp37-cp37m-win_amd64.whl</a></td>
    </tr>
    <tr>
        <td> win_cuda10_cudnn7_mkl </td>
        <td> - </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.0.0/win-mkl/paddlepaddle_gpu-2.0.0a0-cp27-cp27m-win_amd64.whl">
        paddlepaddle_gpu-2.0.0a0-cp27-cp27m-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.0.0/win-mkl/paddlepaddle_gpu-2.0.0a0-cp35-cp35m-win_amd64.whl">
        paddlepaddle_gpu-2.0.0a0-cp35-cp35m-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.0.0/win-mkl/paddlepaddle_gpu-2.0.0a0-cp36-cp36m-win_amd64.whl">
        paddlepaddle_gpu-2.0.0a0-cp36-cp36m-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.0.0/win-mkl/paddlepaddle_gpu-2.0.0a0-cp37-cp37m-win_amd64.whl">
        paddlepaddle_gpu-2.0.0a0-cp37-cp37m-win_amd64.whl</a></td>
    </tr>
    <tr>
        <td> mac_cpu </td>
        <td> - </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.0.0-alpha0-cpu-mac/paddlepaddle-2.0.0a0-cp27-cp27m-macosx_10_6_intel.whl">
        paddlepaddle-2.0.0a0-cp27-cp27m-macosx_10_6_intel.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.0.0-alpha0-cpu-mac/paddlepaddle-2.0.0a0-cp35-cp35m-macosx_10_6_intel.whl">
        paddlepaddle-2.0.0a0-cp35-cp35m-macosx_10_6_intel.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.0.0-alpha0-cpu-mac/paddlepaddle-2.0.0a0-cp36-cp36m-macosx_10_6_intel.whl">
        paddlepaddle-2.0.0a0-cp36-cp36m-macosx_10_6_intel.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.0.0-alpha0-cpu-mac/paddlepaddle-2.0.0a0-cp37-cp37m-macosx_10_6_intel.whl">
        paddlepaddle-2.0.0a0-cp37-cp37m-macosx_10_6_intel.whl</a></td>
    </tr>
    </tbody>
</table>
</p>


### Table instruction

- Vertical axis

cpu-mkl: Support CPU training and prediction, use Intel MKL math library

cuda10_cudnn7-mkl: Support GPU training and prediction, use Intel MKL math library


- Transverse axis

Generally, it is similar to "cp27-cp27mu", in which:

27:python tag, refers to python2. Similarly, there are "35", "36", "37", etc

mu:refers to unicode version python, if it is m, refers to non Unicode version Python

- Installation package naming rules

Each installation package has a unique name. They are named according to the official rules of Python. The form is as follows:

{distribution}-{version}(-{build tag})?-{python tag}-{abi tag}-{platform tag}.whl

The build tag can be missing, and other parts cannot be missing

distribution: wheel name

version: Version, for example 0.14.0 (must be in numeric format)

python tag: similar to 'py27', 'py2', 'py3', used to indicate the corresponding Python version

abi tag: similar to 'cp33m', 'abi3', 'none'

platform tag: similar to 'linux_x86_64', 'any'


<a name="ciwhls"></a>
</br></br>
## **Multi-version whl package list - dev**
<p align="center">
<table>
    <thead>
    <tr>
        <th> version number </th>
        <th> cp27-cp27mu </th>
        <th> cp27-cp27m </th>
        <th> cp35-cp35m    </th>
        <th> cp36-cp36m    </th>
        <th> cp37-cp37m    </th>
    </tr>
    </thead>
    <tbody>
    <tr>
        <td> cpu-mkl </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/0.0.0-cpu-mkl/paddlepaddle-0.0.0-cp27-cp27mu-linux_x86_64.whl">
        paddlepaddle-latest-cp27-cp27mu-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/0.0.0-cpu-mkl/paddlepaddle-0.0.0-cp27-cp27m-linux_x86_64.whl">
        paddlepaddle-latest-cp27-cp27m-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/0.0.0-cpu-mkl/paddlepaddle-0.0.0-cp35-cp35m-linux_x86_64.whl">
        paddlepaddle-latest-cp35-cp35m-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/0.0.0-cpu-mkl/paddlepaddle-0.0.0-cp36-cp36m-linux_x86_64.whl">
        paddlepaddle-latest-cp36-cp36m-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/0.0.0-cpu-mkl/paddlepaddle-0.0.0-cp37-cp37m-linux_x86_64.whl">
        paddlepaddle-latest-cp37-cp37m-linux_x86_64.whl</a></td>
    </tr>
    <tr>
        <td> cpu-openblas </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/0.0.0-cpu-openblas/paddlepaddle-0.0.0-cp27-cp27mu-linux_x86_64.whl">
        paddlepaddle-latest-cp27-cp27mu-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/0.0.0-cpu-openblas/paddlepaddle-0.0.0-cp27-cp27m-linux_x86_64.whl"> paddlepaddle-latest-cp27-cp27m-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/0.0.0-cpu-openblas/paddlepaddle-0.0.0-cp35-cp35m-linux_x86_64.whl">
        paddlepaddle-latest-cp35-cp35m-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/0.0.0-cpu-openblas/paddlepaddle-0.0.0-cp36-cp36m-linux_x86_64.whl">
        paddlepaddle-latest-cp36-cp36m-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/0.0.0-cpu-openblas/paddlepaddle-0.0.0-cp37-cp37m-linux_x86_64.whl">
        paddlepaddle-latest-cp37-cp37m-linux_x86_64.whl</a></td>
    </tr>
    <tr>
        <td> cuda9-cudnn7-openblas </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/0.0.0-gpu-cuda9-cudnn7-openblas/paddlepaddle_gpu-0.0.0-cp27-cp27mu-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp27-cp27mu-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/0.0.0-gpu-cuda9-cudnn7-openblas/paddlepaddle_gpu-0.0.0-cp27-cp27m-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp27-cp27m-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/0.0.0-gpu-cuda9-cudnn7-openblas/paddlepaddle_gpu-0.0.0-cp35-cp35m-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp35-cp35m-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/0.0.0-gpu-cuda9-cudnn7-openblas/paddlepaddle_gpu-0.0.0-cp36-cp36m-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp36-cp36m-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/0.0.0-gpu-cuda9-cudnn7-openblas/paddlepaddle_gpu-0.0.0-cp37-cp37m-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp37-cp37m-linux_x86_64.whl</a></td>
    </tr>
    <tr>
        <td> cuda9-cudnn7-mkl </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/0.0.0-gpu-cuda9-cudnn7-mkl/paddlepaddle_gpu-0.0.0-cp27-cp27mu-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp27-cp27mu-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/0.0.0-gpu-cuda9-cudnn7-mkl/paddlepaddle_gpu-0.0.0-cp27-cp27m-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp27-cp27m-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/0.0.0-gpu-cuda9-cudnn7-mkl/paddlepaddle_gpu-0.0.0-cp35-cp35m-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp35-cp35m-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/0.0.0-gpu-cuda9-cudnn7-mkl/paddlepaddle_gpu-0.0.0-cp36-cp36m-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp36-cp36m-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/0.0.0-gpu-cuda9-cudnn7-mkl/paddlepaddle_gpu-0.0.0-cp37-cp37m-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp37-cp37m-linux_x86_64.whl</a></td>
    </tr>
    <tr>
        <td> cuda10-cudnn7-mkl </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/0.0.0-gpu-cuda10-cudnn7-mkl/paddlepaddle_gpu-0.0.0-cp27-cp27mu-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp27-cp27mu-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/0.0.0-gpu-cuda10-cudnn7-mkl/paddlepaddle_gpu-0.0.0-cp27-cp27m-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp27-cp27m-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/0.0.0-gpu-cuda10-cudnn7-mkl/paddlepaddle_gpu-0.0.0-cp35-cp35m-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp35-cp35m-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/0.0.0-gpu-cuda10-cudnn7-mkl/paddlepaddle_gpu-0.0.0-cp36-cp36m-linux_x86_64.whl">
        paddlepaddle_gpu-latest-cp36-cp36m-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/0.0.0-gpu-cuda10-cudnn7-mkl/paddlepaddle_gpu-0.0.0-cp37-cp37m-linux_x86_64.whl">
        paddlepaddle_gpu-latest-cp37-cp37m-linux_x86_64.whl</a></td>
    </tr>
   </tbody>
</table>
</p>


<a name="ciwhls-gcc8.2-develop"></a>
</br></br>
## **Multi-version whl package list(gcc8.2)-develop**
<p align="center">
<table>
    <thead>
    <tr>
        <th> 版本说明 </th>
        <th> cp27-cp27mu </th>
        <th> cp27-cp27m </th>
        <th> cp35-cp35m    </th>
        <th> cp36-cp36m    </th>
        <th> cp37-cp37m    </th>
    </tr>
    </thead>
    <tbody>
    <tr>
        <td> cuda10.1-cudnn7-mkl </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/0.0.0-gpu-cuda10.1-cudnn7-mkl_gcc8.2/paddlepaddle_gpu-0.0.0-cp27-cp27mu-linux_x86_64.whl">
        paddlepaddle_gpu-0.0.0-cp27-cp27mu-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/0.0.0-gpu-cuda10.1-cudnn7-mkl_gcc8.2/paddlepaddle_gpu-0.0.0-cp27-cp27m-linux_x86_64.whl">
        paddlepaddle_gpu-0.0.0-cp27-cp27m-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/0.0.0-gpu-cuda10.1-cudnn7-mkl_gcc8.2/paddlepaddle_gpu-0.0.0-cp35-cp35m-linux_x86_64.whl">
        paddlepaddle_gpu-0.0.0-cp35-cp35m-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/0.0.0-gpu-cuda10.1-cudnn7-mkl_gcc8.2/paddlepaddle_gpu-0.0.0-cp36-cp36m-linux_x86_64.whl">
        paddlepaddle_gpu-0.0.0-cp36-cp36m-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/0.0.0-gpu-cuda10.1-cudnn7-mkl_gcc8.2/paddlepaddle_gpu-0.0.0-cp37-cp37m-linux_x86_64.whl">
        paddlepaddle_gpu-0.0.0-cp37-cp37m-linux_x86_64.whl</a></td>
    </tr>
   </tbody>
</table>
</p>


<!--TODO this part should be in a new webpage-->

</br></br>


## Execute the PaddlePaddle training program in Docker


Suppose you have written a PaddlePaddle program in the current directory (such as /home/work): `train.py` ( refer to [PaddlePaddleBook](https://github.com/PaddlePaddle/book/blob/develop/01.fit_a_line/README.cn.md) to write), you can start the training with the following command:


    cd /home/work
    docker run -it -v $PWD:/work hub.baidubce.com/paddlepaddle/paddle /work/train.py


In the above commands, the `-it` parameter indicates that the container has been run interactively; `-v $PWD:/work` specifies that the current path (the absolute path where the PWD variable in Linux will expand to the current path) is mounted to the `:/work` directory inside the container: `Hub.baidubce.com/paddlepaddle/paddle` specifies the container to be used; finally `/work/train.py` is the command executed inside the container, ie. the training program.

Of course, you can also enter into the Docker container and execute or debug your code interactively:


    docker run -it -v $PWD:/work hub.baidubce.com/paddlepaddle/paddle /bin/bash
    cd /work
    python train.py


**Note: In order to reduce the size, vim is not installed in PaddlePaddle Docker image by default. You can edit the code in the container after executing ** `apt-get install -y vim` **(which installs vim for you) in the container.**

</br></br>

## Start PaddlePaddle Book tutorial with Docker


Use Docker to quickly launch a local Jupyter Notebook containing the PaddlePaddle official Book tutorial, which can be viewed on the web. PaddlePaddle Book is an interactive Jupyter Notebook for users and developers. If you want to learn more about deep learning, PaddlePaddle Book is definitely your best choice. You can read tutorials or create and share interactive documents with code, formulas, charts, and text.

We provide a Docker image that can run the PaddlePaddle Book directly, running directly:

`docker run -p 8888:8888 hub.baidubce.com/paddlepaddle/book`

Domestic users can use the following image source to speed up access:

`docker run -p 8888:8888 hub.baidubce.com/paddlepaddle/book`

Then enter the following URL in your browser:

`http://localhost:8888/`

It's that simple and bon voyage! For further questions, please refer to the [FAQ](#FAQ).


</br></br>
## Perform GPU training using Docker


In order to ensure that the GPU driver works properly in the image, we recommend using [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) to run the image. Don't forget to install the latest GPU drivers on your physical machine in advance.

`Nvidia-docker run -it -v $PWD:/work hub.baidubce.com/paddlepaddle/paddle:latest-gpu /bin/bash`

**Note: If you don't have nvidia-docker installed, you can try the following to mount the CUDA library and Linux devices into the Docker container:**


    export CUDA_SO="$(\ls /usr/lib64/libcuda* | xargs -I{} echo '-v {}:{}') \
    $(\ls /usr/lib64/libnvidia* | xargs -I{} echo '-v {}:{}')"
    export DEVICES=$(\ls /dev/nvidia* | xargs -I{} echo '--device {}:{}')
    docker run ${CUDA_SO} \
      ${DEVICES} -it hub.baidubce.com/paddlepaddle/paddle:latest-gpu
