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
        <td> 3.15, 3.16(Recommend),3.17 </td>
        <td>  </td>
        <td>  </td>
    </tr>
    <tr>
        <td> GCC </td>
        <td> 5.4 / 8.2 </td>
        <td>  recommends using devtools2 for CentOS </td>
        <td>  </td>
    </tr>
    <tr>
        <td> Clang (macOS Only) </td>
        <td> 9.0 and above </td>
        <td> Usually use the clang version of macOS 10.11 and above </td>
        <td>  </td>
    </tr>
        <tr>
        <td> Python（64 bit） </td>
        <td> 3.7+ </td>
        <td> depends on libpython3.7+.so </td>
        <td> please go to <a href="https://www.python.org">Python official website </a></td>
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
        <td> at least 20.2.2 </td>
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
        <td> brew install rar (For macOS), apt-get install unrar (For Ubuntu) </td>
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

> `cmake .. -DWITH_GPU=ON -DWITH_TESTING=OFF -DCUDNN_ROOT=/opt/cudnnv5`

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
        <td> paddlepaddle==[version code] such as paddlepaddle==2.2.1 </td>
        <td> Only support the corresponding version of the CPU PaddlePaddle, please refer to <a href=https://pypi.org/project/paddlepaddle/#history>Pypi</a> for the specific version. </td>
    </tr>
    <tr>
        <td> paddlepaddle-gpu==[version code], such as paddlepaddle-gpu==2.2.1 </td>
        <td> The default installation supports the PaddlePaddle installation package corresponding to [version number] of CUDA 10.2 and cuDNN 7 </td>
    </tr>
   </tbody>
</table>
</p>

You can find various distributions of PaddlePaddle-gpu in [the Release History](https://pypi.org/project/paddlepaddle-gpu/#history).
> 'postxx' corresponds to CUDA and cuDNN versions, and the number before 'postxx' represents the version of Paddle

Please note that: in the commands, <code> paddlepaddle-gpu==2.2.1 </code> will install the installation package of PaddlePaddle that supports CUDA 10.2 and cuDNN 7 by default under Windows environment.


<a name="ciwhls-release"></a>
</br></br>

## **Multi-version whl package list - Release**


<p align="center">
<table>
    <thead>
    <tr>
        <th> Release Instruction </th>
        <th> cp36-cp36m    </th>
        <th> cp37-cp37m    </th>
        <th> cp38-cp38    </th>
        <th> cp39-cp39    </th>
    </tr>
    </thead>
    <tbody>
    <tr>
        <td> cpu-mkl-avx </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.2.1/linux/linux-cpu-mkl-avx/paddlepaddle-2.2.1-cp36-cp36m-linux_x86_64.whl"> paddlepaddle-2.2.1-cp36-cp36m-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.2.1/linux/linux-cpu-mkl-avx/paddlepaddle-2.2.1-cp37-cp37m-linux_x86_64.whl"> paddlepaddle-2.2.1-cp37-cp37m-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.2.1/linux/linux-cpu-mkl-avx/paddlepaddle-2.2.1-cp38-cp38-linux_x86_64.whl"> paddlepaddle-2.2.1-cp38-cp38-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.2.1/linux/linux-cpu-mkl-avx/paddlepaddle-2.2.1-cp39-cp39-linux_x86_64.whl"> paddlepaddle-2.2.1-cp39-cp39-linux_x86_64.whl</a></td>
    </tr>
    <tr>
        <td> cpu-openblas-avx </td>
        <td> - </td>
        <td> - </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.2.1/linux/linux-cpu-openblas-avx/paddlepaddle-2.2.1-cp38-cp38-linux_x86_64.whl"> paddlepaddle-2.2.1-cp38-cp38-linux_x86_64.whl</a></td>
        <td> - </td>
    </tr>
    <tr>
        <td> cpu-mkl-noavx </td>
        <td> - </td>
        <td> - </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.2.1/linux/linux-cpu-mkl-noavx/paddlepaddle-2.2.1-cp38-cp38-linux_x86_64.whl"> paddlepaddle-2.2.1-cp38-cp38-linux_x86_64.whl</a></td>
        <td> - </td>
    </tr>
    <tr>
        <td> cpu-openblas-noavx </td>
        <td> - </td>
        <td> - </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.2.1/linux/linux-cpu-openblas-noavx/paddlepaddle-2.2.1-cp38-cp38-linux_x86_64.whl"> paddlepaddle-2.2.1-cp38-cp38-linux_x86_64.whl</a></td>
        <td> - </td>
    </tr>
    <tr>
        <td> cuda10.1-cudnn7-mkl-gcc5.4-avx </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.2.1/linux/linux-gpu-cuda10.1-cudnn7-mkl-gcc5.4-avx/paddlepaddle_gpu-2.2.1.post101-cp36-cp36m-linux_x86_64.whl">
        paddlepaddle_gpu-2.2.1.post101-cp36-cp36m-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.2.1/linux/linux-gpu-cuda10.1-cudnn7-mkl-gcc5.4-avx/paddlepaddle_gpu-2.2.1.post101-cp37-cp37m-linux_x86_64.whl">
        paddlepaddle_gpu-2.2.1.post101-cp37-cp37m-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.2.1/linux/linux-gpu-cuda10.1-cudnn7-mkl-gcc5.4-avx/paddlepaddle_gpu-2.2.1.post101-cp38-cp38-linux_x86_64.whl">
        paddlepaddle_gpu-2.2.1.post101-cp38-cp38-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.2.1/linux/linux-gpu-cuda10.1-cudnn7-mkl-gcc5.4-avx/paddlepaddle_gpu-2.2.1.post101-cp39-cp39-linux_x86_64.whl">
        paddlepaddle_gpu-2.2.1.post101-cp39-cp39-linux_x86_64.whl</a></td>
    </tr>
    <tr>
        <td> cuda10.1-cudnn7-mkl-gcc5.4-noavx </td>
        <td> - </td>
        <td> - </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.2.1/linux/linux-gpu-cuda10.1-cudnn7-mkl-gcc5.4-noavx/paddlepaddle_gpu-2.2.1.post101-cp38-cp38-linux_x86_64.whl">
        paddlepaddle_gpu-2.2.1.post101-cp38-cp38-linux_x86_64.whl</a></td>
        <td> - </td>
    </tr>
    <tr>
        <td> cuda10.2-cudnn7-mkl-gcc8.2-avx </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.2.1/linux/linux-gpu-cuda10.2-cudnn7-mkl-gcc8.2-avx/paddlepaddle_gpu-2.2.1-cp36-cp36m-linux_x86_64.whl">
        paddlepaddle_gpu-2.2.1-cp36-cp36m-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.2.1/linux/linux-gpu-cuda10.2-cudnn7-mkl-gcc8.2-avx/paddlepaddle_gpu-2.2.1-cp37-cp37m-linux_x86_64.whl">
        paddlepaddle_gpu-2.2.1-cp37-cp37m-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.2.1/linux/linux-gpu-cuda10.2-cudnn7-mkl-gcc8.2-avx/paddlepaddle_gpu-2.2.1-cp38-cp38-linux_x86_64.whl">
        paddlepaddle_gpu-2.2.1-cp38-cp38-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.2.1/linux/linux-gpu-cuda10.2-cudnn7-mkl-gcc8.2-avx/paddlepaddle_gpu-2.2.1-cp39-cp39-linux_x86_64.whl">
        paddlepaddle_gpu-2.2.1-cp39-cp39-linux_x86_64.whl</a></td>
    </tr>
    <tr>
        <td> cuda10.2-cudnn7-mkl-gcc8.2-noavx </td>
        <td> - </td>
        <td> - </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.2.1/linux/linux-gpu-cuda10.2-cudnn7-mkl-gcc8.2-noavx/paddlepaddle_gpu-2.2.1-cp38-cp38-linux_x86_64.whl">
        paddlepaddle_gpu-2.2.1-cp38-cp38-linux_x86_64.whl</a></td>
        <td> - </td>
    </tr>
    <tr>
        <td> cuda11.0-cudnn8.0-mkl-gcc8.2-avx </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.2.1/linux/linux-gpu-cuda11.0-cudnn8-mkl-gcc8.2-avx/paddlepaddle_gpu-2.2.1.post110-cp36-cp36m-linux_x86_64.whl">
        paddlepaddle_gpu-2.2.1.post110-cp36-cp36m-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.2.1/linux/linux-gpu-cuda11.0-cudnn8-mkl-gcc8.2-avx/paddlepaddle_gpu-2.2.1.post110-cp37-cp37m-linux_x86_64.whl">
        paddlepaddle_gpu-2.2.1.post110-cp37-cp37m-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.2.1/linux/linux-gpu-cuda11.0-cudnn8-mkl-gcc8.2-avx/paddlepaddle_gpu-2.2.1.post110-cp38-cp38-linux_x86_64.whl">
        paddlepaddle_gpu-2.2.1.post110-cp38-cp38-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.2.1/linux/linux-gpu-cuda11.0-cudnn8-mkl-gcc8.2-avx/paddlepaddle_gpu-2.2.1.post110-cp39-cp39-linux_x86_64.whl">
        paddlepaddle_gpu-2.2.1.post110-cp39-cp39-linux_x86_64.whl</a></td>
    </tr>
    <tr>
        <td> cuda11.1-cudnn8.1-mkl-gcc8.2-avx </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.2.1/linux/linux-gpu-cuda11.1-cudnn8.1-mkl-gcc8.2-avx/paddlepaddle_gpu-2.2.1.post111-cp36-cp36m-linux_x86_64.whl">
        paddlepaddle_gpu-2.2.1.post111-cp36-cp36m-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.2.1/linux/linux-gpu-cuda11.1-cudnn8.1-mkl-gcc8.2-avx/paddlepaddle_gpu-2.2.1.post111-cp37-cp37m-linux_x86_64.whl">
        paddlepaddle_gpu-2.2.1.post111-cp37-cp37m-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.2.1/linux/linux-gpu-cuda11.1-cudnn8.1-mkl-gcc8.2-avx/paddlepaddle_gpu-2.2.1.post111-cp38-cp38-linux_x86_64.whl">
        paddlepaddle_gpu-2.2.1.post111-cp38-cp38-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.2.1/linux/linux-gpu-cuda11.1-cudnn8.1-mkl-gcc8.2-avx/paddlepaddle_gpu-2.2.1.post111-cp39-cp39-linux_x86_64.whl">
        paddlepaddle_gpu-2.2.1.post111-cp39-cp39-linux_x86_64.whl</a></td>
    </tr>
    <tr>
        <td> cuda11.2-cudnn8.1-mkl-gcc8.2-avx </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.2.1/linux/linux-gpu-cuda11.2-cudnn8-mkl-gcc8.2-avx/paddlepaddle_gpu-2.2.1.post112-cp36-cp36m-linux_x86_64.whl">
        paddlepaddle_gpu-2.2.1.post112-cp36-cp36m-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.2.1/linux/linux-gpu-cuda11.2-cudnn8-mkl-gcc8.2-avx/paddlepaddle_gpu-2.2.1.post112-cp37-cp37m-linux_x86_64.whl">
        paddlepaddle_gpu-2.2.1.post112-cp37-cp37m-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.2.1/linux/linux-gpu-cuda11.2-cudnn8-mkl-gcc8.2-avx/paddlepaddle_gpu-2.2.1.post112-cp38-cp38-linux_x86_64.whl">
        paddlepaddle_gpu-2.2.1.post112-cp38-cp38-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.2.1/linux/linux-gpu-cuda11.2-cudnn8-mkl-gcc8.2-avx/paddlepaddle_gpu-2.2.1.post112-cp39-cp39-linux_x86_64.whl">
        paddlepaddle_gpu-2.2.1.post112-cp39-cp39-linux_x86_64.whl</a></td>
    </tr>
    <tr>
        <td> macos-cpu-openblas </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.2.1/macos/macos-cpu-openblas/paddlepaddle-2.2.1-cp36-cp36m-macosx_10_6_intel.whl">
        paddlepaddle-2.2.1-cp36-cp36m-macosx_10_6_intel.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.2.1/macos/macos-cpu-openblas/paddlepaddle-2.2.1-cp37-cp37m-macosx_10_6_intel.whl">
        paddlepaddle-2.2.1-cp37-cp37m-macosx_10_6_intel.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.2.1/macos/macos-cpu-openblas/paddlepaddle-2.2.1-cp38-cp38-macosx_10_14_x86_64.whl">
        paddlepaddle-2.2.1-cp38-cp38-macosx_10_14_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.2.1/macos/macos-cpu-openblas/paddlepaddle-2.2.1-cp39-cp39-macosx_10_14_x86_64.whl">
        paddlepaddle-2.2.1-cp39-cp39-macosx_10_14_x86_64.whl</a></td>
    </tr>
    <tr>
        <td> win-cpu-mkl-avx </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.2.1/windows/windows-cpu-mkl-avx/paddlepaddle-2.2.1-cp36-cp36m-win_amd64.whl"> paddlepaddle-2.2.1-cp36-cp36m-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.2.1/windows/windows-cpu-mkl-avx/paddlepaddle-2.2.1-cp37-cp37m-win_amd64.whl"> paddlepaddle-2.2.1-cp37-cp37m-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.2.1/windows/windows-cpu-mkl-avx/paddlepaddle-2.2.1-cp38-cp38-win_amd64.whl"> paddlepaddle-2.2.1-cp38-cp38-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.2.1/windows/windows-cpu-mkl-avx/paddlepaddle-2.2.1-cp39-cp39-win_amd64.whl"> paddlepaddle-2.2.1-cp39-cp39-win_amd64.whl</a></td>
    </tr>
    <tr>
        <td> win-cpu-mkl-noavx </td>
        <td> - </td>
        <td> - </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.2.1/windows/windows-cpu-mkl-noavx/paddlepaddle-2.2.1-cp38-cp38-win_amd64.whl"> paddlepaddle-2.2.1-cp38-cp38-win_amd64.whl</a></td>
        <td> - </td>
    </tr>
    <tr>
        <td> win-cpu-openblas-avx </td>
        <td> - </td>
        <td> - </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.2.1/windows/windows-cpu-openblas-avx/paddlepaddle-2.2.1-cp38-cp38-win_amd64.whl"> paddlepaddle-2.2.1-cp38-cp38-win_amd64.whl</a></td>
        <td> - </td>
    </tr>
    <tr>
        <td> win-cpu-openblas-noavx </td>
        <td> - </td>
        <td> - </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.2.1/windows/windows-cpu-openblas-noavx/paddlepaddle-2.2.1-cp38-cp38-win_amd64.whl"> paddlepaddle-2.2.1-cp38-cp38-win_amd64.whl</a></td>
        <td> - </td>
    </tr>
    <tr>
        <td> win-cuda10.1-cudnn7-mkl-vs2017-avx </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.2.1/windows/windows-gpu-cuda10.1-cudnn7-mkl-avx/paddlepaddle_gpu-2.2.1.post101-cp36-cp36m-win_amd64.whl"> paddlepaddle_gpu-2.2.1.post101-cp36-cp36m-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.2.1/windows/windows-gpu-cuda10.1-cudnn7-mkl-avx/paddlepaddle_gpu-2.2.1.post101-cp37-cp37m-win_amd64.whl"> paddlepaddle_gpu-2.2.1.post101-cp37-cp37m-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.2.1/windows/windows-gpu-cuda10.1-cudnn7-mkl-avx/paddlepaddle_gpu-2.2.1.post101-cp38-cp38-win_amd64.whl"> paddlepaddle_gpu-2.2.1.post101-cp38-cp38-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.2.1/windows/windows-gpu-cuda10.1-cudnn7-mkl-avx/paddlepaddle_gpu-2.2.1.post101-cp39-cp39-win_amd64.whl"> paddlepaddle_gpu-2.2.1.post101-cp39-cp39-win_amd64.whl</a></td>
    </tr>
    <tr>
        <td> win-cuda10.1-cudnn7-mkl-vs2017-noavx </td>
        <td> - </td>
        <td> - </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.2.1/windows/windows-gpu-cuda10.1-cudnn7-mkl-noavx/paddlepaddle_gpu-2.2.1.post101-cp38-cp38-win_amd64.whl"> paddlepaddle_gpu-2.2.1.post101-cp38-cp38-win_amd64.whl</a></td>
        <td> - </td>
    </tr>
    <tr>
        <td> win-cuda10.2-cudnn7-mkl-vs2017-avx </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.2.1/windows/windows-gpu-cuda10.2-cudnn7-mkl-avx/paddlepaddle_gpu-2.2.1-cp36-cp36m-win_amd64.whl"> paddlepaddle_gpu-2.2.1.post102-cp36-cp36m-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.2.1/windows/windows-gpu-cuda10.2-cudnn7-mkl-avx/paddlepaddle_gpu-2.2.1-cp37-cp37m-win_amd64.whl"> paddlepaddle_gpu-2.2.1.post102-cp37-cp37m-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.2.1/windows/windows-gpu-cuda10.2-cudnn7-mkl-avx/paddlepaddle_gpu-2.2.1-cp38-cp38-win_amd64.whl"> paddlepaddle_gpu-2.2.1.post102-cp38-cp38-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.2.1/windows/windows-gpu-cuda10.2-cudnn7-mkl-avx/paddlepaddle_gpu-2.2.1-cp39-cp39-win_amd64.whl"> paddlepaddle_gpu-2.2.1.post102-cp39-cp39-win_amd64.whl</a></td>
    </tr>
    <tr>
        <td> win-cuda10.2-cudnn7-mkl-vs2017-noavx </td>
        <td> - </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.2.1/windows/windows-gpu-cuda10.2-cudnn7-mkl-noavx/paddlepaddle_gpu-2.2.1-cp37-cp37m-win_amd64.whl"> paddlepaddle_gpu-2.2.1.post102-cp37-cp37m-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.2.1/windows/windows-gpu-cuda10.2-cudnn7-mkl-noavx/paddlepaddle_gpu-2.2.1-cp38-cp38-win_amd64.whl"> paddlepaddle_gpu-2.2.1.post102-cp38-cp38-win_amd64.whl</a></td>
        <td> - </td>
    </tr>
    <tr>
        <td> win-cuda11.0-cudnn8.0-mkl-vs2017-avx </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.2.1/windows/windows-gpu-cuda11.0-cudnn8-mkl-avx/paddlepaddle_gpu-2.2.1.post110-cp36-cp36m-win_amd64.whl"> paddlepaddle_gpu-2.2.1.post110-cp36-cp36m-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.2.1/windows/windows-gpu-cuda11.0-cudnn8-mkl-avx/paddlepaddle_gpu-2.2.1.post110-cp37-cp37m-win_amd64.whl"> paddlepaddle_gpu-2.2.1.post110-cp37-cp37m-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.2.1/windows/windows-gpu-cuda11.0-cudnn8-mkl-avx/paddlepaddle_gpu-2.2.1.post110-cp38-cp38-win_amd64.whl"> paddlepaddle_gpu-2.2.1.post110-cp38-cp38-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.2.1/windows/windows-gpu-cuda11.0-cudnn8-mkl-avx/paddlepaddle_gpu-2.2.1.post110-cp39-cp39-win_amd64.whl"> paddlepaddle_gpu-2.2.1.post110-cp39-cp39-win_amd64.whl</a></td>
    </tr>
    <tr>
        <td> win-cuda11.1-cudnn8.1-mkl-vs2017-avx </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.2.1/windows/windows-gpu-cuda11.1-cudnn8-mkl-avx/paddlepaddle_gpu-2.2.1.post111-cp36-cp36m-win_amd64.whl"> paddlepaddle_gpu-2.2.1.post111-cp36-cp36m-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.2.1/windows/windows-gpu-cuda11.1-cudnn8-mkl-avx/paddlepaddle_gpu-2.2.1.post111-cp37-cp37m-win_amd64.whl"> paddlepaddle_gpu-2.2.1.post111-cp37-cp37m-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.2.1/windows/windows-gpu-cuda11.1-cudnn8-mkl-avx/paddlepaddle_gpu-2.2.1.post111-cp38-cp38-win_amd64.whl"> paddlepaddle_gpu-2.2.1.post111-cp38-cp38-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.2.1/windows/windows-gpu-cuda11.1-cudnn8-mkl-avx/paddlepaddle_gpu-2.2.1.post111-cp39-cp39-win_amd64.whl"> paddlepaddle_gpu-2.2.1.post111-cp39-cp39-win_amd64.whl</a></td>
    </tr>
    <tr>
        <td> win-cuda11.2-cudnn8.2-mkl-vs2017-avx </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.2.1/windows/windows-gpu-cuda11.2-cudnn8-mkl-avx/paddlepaddle_gpu-2.2.1.post112-cp36-cp36m-win_amd64.whl"> paddlepaddle_gpu-2.2.1.post112-cp36-cp36m-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.2.1/windows/windows-gpu-cuda11.2-cudnn8-mkl-avx/paddlepaddle_gpu-2.2.1.post112-cp37-cp37m-win_amd64.whl"> paddlepaddle_gpu-2.2.1.post112-cp37-cp37m-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.2.1/windows/windows-gpu-cuda11.2-cudnn8-mkl-avx/paddlepaddle_gpu-2.2.1.post112-cp38-cp38-win_amd64.whl"> paddlepaddle_gpu-2.2.1.post112-cp38-cp38-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.2.1/windows/windows-gpu-cuda11.2-cudnn8-mkl-avx/paddlepaddle_gpu-2.2.1.post112-cp39-cp39-win_amd64.whl"> paddlepaddle_gpu-2.2.1.post112-cp39-cp39-win_amd64.whl</a></td>
    </tr>
    </tbody>
</table>
</p>

## **Multi-version whl package list - Release/llm_2.5**

<p align="center">
<table>
    <thead>
    <tr>
        <th> Release Instruction </th>
        <th> cp37-cp37m    </th>
        <th> cp38-cp38    </th>
        <th> cp39-cp39    </th>
        <th> cp310-cp310    </th>
        <th> cp311-cp311    </th>
    </tr>
    </thead>
    <tbody>
    <tr>
        <td> linux-cuda11.8-cudnn8.6-trt8.5-mkl-gcc8.2-avx-llm-volta </td>
        <td> <a href="https://paddle-qa.bj.bcebos.com/paddle-pipeline/LLM-Linux-Gpu-Cuda11.8-Cudnn8.6-Trt8.5-Mkl-Avx-Gcc8.2/latest/paddlepaddle_gpu-2.6.0.dev0%2Bcu118.volta.llm-cp37-cp37m-linux_x86_64.whl">paddlepaddle_gpu-2.6.0.dev0+cu118.volta.llm-cp37-cp37m-linux_x86_64.whl</a> </td>
        <td> <a href="https://paddle-qa.bj.bcebos.com/paddle-pipeline/LLM-Linux-Gpu-Cuda11.8-Cudnn8.6-Trt8.5-Mkl-Avx-Gcc8.2/latest/paddlepaddle_gpu-2.6.0.dev0%2Bcu118.volta.llm-cp38-cp38-linux_x86_64.whl">paddlepaddle_gpu-2.6.0.dev0+cu118.volta.llm-cp38-cp38-linux_x86_64.whl</a> </td>
        <td> <a href="https://paddle-qa.bj.bcebos.com/paddle-pipeline/LLM-Linux-Gpu-Cuda11.8-Cudnn8.6-Trt8.5-Mkl-Avx-Gcc8.2/latest/paddlepaddle_gpu-2.6.0.dev0%2Bcu118.volta.llm-cp39-cp39-linux_x86_64.whl">paddlepaddle_gpu-2.6.0.dev0+cu118.volta.llm-cp39-cp39-linux_x86_64.whl</a> </td>
        <td> <a href="https://paddle-qa.bj.bcebos.com/paddle-pipeline/LLM-Linux-Gpu-Cuda11.8-Cudnn8.6-Trt8.5-Mkl-Avx-Gcc8.2/latest/paddlepaddle_gpu-2.6.0.dev0%2Bcu118.volta.llm-cp310-cp310-linux_x86_64.whl">paddlepaddle_gpu-2.6.0.dev0+cu118.volta.llm-cp310-cp310-linux_x86_64.whl</a> </td>
        <td> <a href="https://paddle-qa.bj.bcebos.com/paddle-pipeline/LLM-Linux-Gpu-Cuda11.8-Cudnn8.6-Trt8.5-Mkl-Avx-Gcc8.2/latest/paddlepaddle_gpu-2.6.0.dev0%2Bcu118.volta.llm-cp311-cp311-linux_x86_64.whl">paddlepaddle_gpu-2.6.0.dev0+cu118.volta.llm-cp311-cp311-linux_x86_64.whl</a> </td>
    </tr>
    <tr>
        <td> linux-cuda11.8-cudnn8.6-trt8.5-mkl-gcc8.2-avx-llm-ampere </td>
        <td> <a href="https://paddle-qa.bj.bcebos.com/paddle-pipeline/LLM-Linux-Gpu-Cuda11.8-Cudnn8.6-Trt8.5-Mkl-Avx-Gcc8.2/latest/paddlepaddle_gpu-2.6.0.dev0%2Bcu118.ampere.llm-cp37-cp37m-linux_x86_64.whl">paddlepaddle_gpu-2.6.0.dev0+cu118.ampere.llm-cp37-cp37m-linux_x86_64.whl</a> </td>
        <td> <a href="https://paddle-qa.bj.bcebos.com/paddle-pipeline/LLM-Linux-Gpu-Cuda11.8-Cudnn8.6-Trt8.5-Mkl-Avx-Gcc8.2/latest/paddlepaddle_gpu-2.6.0.dev0%2Bcu118.ampere.llm-cp38-cp38-linux_x86_64.whl">paddlepaddle_gpu-2.6.0.dev0+cu118.ampere.llm-cp38-cp38-linux_x86_64.whl</a> </td>
        <td> <a href="https://paddle-qa.bj.bcebos.com/paddle-pipeline/LLM-Linux-Gpu-Cuda11.8-Cudnn8.6-Trt8.5-Mkl-Avx-Gcc8.2/latest/paddlepaddle_gpu-2.6.0.dev0%2Bcu118.ampere.llm-cp39-cp39-linux_x86_64.whl">paddlepaddle_gpu-2.6.0.dev0+cu118.ampere.llm-cp39-cp39-linux_x86_64.whl</a> </td>
        <td> <a href="https://paddle-qa.bj.bcebos.com/paddle-pipeline/LLM-Linux-Gpu-Cuda11.8-Cudnn8.6-Trt8.5-Mkl-Avx-Gcc8.2/latest/paddlepaddle_gpu-2.6.0.dev0%2Bcu118.ampere.llm-cp310-cp310-linux_x86_64.whl">paddlepaddle_gpu-2.6.0.dev0+cu118.ampere.llm-cp310-cp310-linux_x86_64.whl</a> </td>
        <td> <a href="https://paddle-qa.bj.bcebos.com/paddle-pipeline/LLM-Linux-Gpu-Cuda11.8-Cudnn8.6-Trt8.5-Mkl-Avx-Gcc8.2/latest/paddlepaddle_gpu-2.6.0.dev0%2Bcu118.ampere.llm-cp311-cp311-linux_x86_64.whl">paddlepaddle_gpu-2.6.0.dev0+cu118.ampere.llm-cp311-cp311-linux_x86_64.whl</a> </td>
    </tr>
    <tr>
        <td>linux-cuda12.0-cudnn8.9-trt8.6-mkl-gcc12.2-avx-llm-volta</td>
        <td> - </td>
        <td> <a href="https://paddle-qa.bj.bcebos.com/paddle-pipeline/LLM-Linux-Gpu-Cuda12.0-Cudnn8.9-Trt8.6-Mkl-Avx-Gcc12.2/latest/paddlepaddle_gpu-2.6.0.dev0%2Bcu120.volta.llm-cp38-cp38-linux_x86_64.whl">paddlepaddle_gpu-2.6.0.dev0+cu120.volta.llm-cp38-cp38-linux_x86_64.whl</a> </td>
        <td> <a href="https://paddle-qa.bj.bcebos.com/paddle-pipeline/LLM-Linux-Gpu-Cuda12.0-Cudnn8.9-Trt8.6-Mkl-Avx-Gcc12.2/latest/paddlepaddle_gpu-2.6.0.dev0%2Bcu120.volta.llm-cp39-cp39-linux_x86_64.whl">paddlepaddle_gpu-2.6.0.dev0+cu120.volta.llm-cp39-cp39-linux_x86_64.whl</a> </td>
        <td> <a href="https://paddle-qa.bj.bcebos.com/paddle-pipeline/LLM-Linux-Gpu-Cuda12.0-Cudnn8.9-Trt8.6-Mkl-Avx-Gcc12.2/latest/paddlepaddle_gpu-2.6.0.dev0%2Bcu120.volta.llm-cp310-cp310-linux_x86_64.whl">paddlepaddle_gpu-2.6.0.dev0+cu120.volta.llm-cp310-cp310-linux_x86_64.whl</a> </td>
        <td> <a href="https://paddle-qa.bj.bcebos.com/paddle-pipeline/LLM-Linux-Gpu-Cuda12.0-Cudnn8.9-Trt8.6-Mkl-Avx-Gcc12.2/latest/paddlepaddle_gpu-2.6.0.dev0%2Bcu120.volta.llm-cp311-cp311-linux_x86_64.whl">paddlepaddle_gpu-2.6.0.dev0+cu120.volta.llm-cp311-cp311-linux_x86_64.whl</a> </td>
    </tr>
    <tr>
        <td>linux-cuda12.0-cudnn8.9-trt8.6-mkl-gcc12.2-avx-llm-ampere</td>
        <td> - </td>
        <td> <a href="https://paddle-qa.bj.bcebos.com/paddle-pipeline/LLM-Linux-Gpu-Cuda12.0-Cudnn8.9-Trt8.6-Mkl-Avx-Gcc12.2/latest/paddlepaddle_gpu-2.6.0.dev0%2Bcu120.ampere.llm-cp38-cp38-linux_x86_64.whl">paddlepaddle_gpu-2.6.0.dev0+cu120.ampere.llm-cp38-cp38-linux_x86_64.whl</a> </td>
        <td> <a href="https://paddle-qa.bj.bcebos.com/paddle-pipeline/LLM-Linux-Gpu-Cuda12.0-Cudnn8.9-Trt8.6-Mkl-Avx-Gcc12.2/latest/paddlepaddle_gpu-2.6.0.dev0%2Bcu120.ampere.llm-cp39-cp39-linux_x86_64.whl">paddlepaddle_gpu-2.6.0.dev0+cu120.ampere.llm-cp39-cp39-linux_x86_64.whl</a> </td>
        <td> <a href="https://paddle-qa.bj.bcebos.com/paddle-pipeline/LLM-Linux-Gpu-Cuda12.0-Cudnn8.9-Trt8.6-Mkl-Avx-Gcc12.2/latest/paddlepaddle_gpu-2.6.0.dev0%2Bcu120.ampere.llm-cp310-cp310-linux_x86_64.whl">paddlepaddle_gpu-2.6.0.dev0+cu120.ampere.llm-cp310-cp310-linux_x86_64.whl</a> </td>
        <td> <a href="https://paddle-qa.bj.bcebos.com/paddle-pipeline/LLM-Linux-Gpu-Cuda12.0-Cudnn8.9-Trt8.6-Mkl-Avx-Gcc12.2/latest/paddlepaddle_gpu-2.6.0.dev0%2Bcu120.ampere.llm-cp311-cp311-linux_x86_64.whl">paddlepaddle_gpu-2.6.0.dev0+cu120.ampere.llm-cp311-cp311-linux_x86_64.whl</a> </td>
    </tr>
    <tr>
        <td> windows-cuda11.8-cudnn8.6-trt8.5-mkl-vs2019-avx-llm-volta </td>
        <td> <a href="https://paddle-qa.bj.bcebos.com/paddle-pipeline/LLM-Windows-Gpu-Cuda11.8-Cudnn8.6-Trt8.5-Mkl-Avx-Vs2019/latest/paddlepaddle_gpu-2.6.0.dev0%2Bcu118.volta.llm-cp37-cp37m-win_amd64.whl">paddlepaddle_gpu-2.6.0.dev0+cu118.volta.llm-cp37-cp37m-win_amd64.whl</a> </td>
        <td> <a href="https://paddle-qa.bj.bcebos.com/paddle-pipeline/LLM-Windows-Gpu-Cuda11.8-Cudnn8.6-Trt8.5-Mkl-Avx-Vs2019/latest/paddlepaddle_gpu-2.6.0.dev0%2Bcu118.volta.llm-cp38-cp38-win_amd64.whl">paddlepaddle_gpu-2.6.0.dev0+cu118.volta.llm-cp38-cp38-win_amd64.whl</a> </td>
        <td> <a href="https://paddle-qa.bj.bcebos.com/paddle-pipeline/LLM-Windows-Gpu-Cuda11.8-Cudnn8.6-Trt8.5-Mkl-Avx-Vs2019/latest/paddlepaddle_gpu-2.6.0.dev0%2Bcu118.volta.llm-cp39-cp39-win_amd64.whl">paddlepaddle_gpu-2.6.0.dev0+cu118.volta.llm-cp39-cp39-win_amd64.whl</a> </td>
        <td> <a href="https://paddle-qa.bj.bcebos.com/paddle-pipeline/LLM-Windows-Gpu-Cuda11.8-Cudnn8.6-Trt8.5-Mkl-Avx-Vs2019/latest/paddlepaddle_gpu-2.6.0.dev0%2Bcu118.volta.llm-cp310-cp310-win_amd64.whl">paddlepaddle_gpu-2.6.0.dev0+cu118.volta.llm-cp310-cp310-win_amd64.whl</a> </td>
        <td> <a href="https://paddle-qa.bj.bcebos.com/paddle-pipeline/LLM-Windows-Gpu-Cuda11.8-Cudnn8.6-Trt8.5-Mkl-Avx-Vs2019/latest/paddlepaddle_gpu-2.6.0.dev0%2Bcu118.volta.llm-cp311-cp311-win_amd64.whl">paddlepaddle_gpu-2.6.0.dev0+cu118.volta.llm-cp311-cp311-win_amd64.whl</a> </td>
    </tr>
    <tr>
        <td> windows-cuda11.8-cudnn8.6-trt8.5-mkl-vs2019-avx-llm-ampere </td>
        <td> <a href="https://paddle-qa.bj.bcebos.com/paddle-pipeline/LLM-Windows-Gpu-Cuda11.8-Cudnn8.6-Trt8.5-Mkl-Avx-Vs2019/latest/paddlepaddle_gpu-2.6.0.dev0%2Bcu118.ampere.llm-cp37-cp37m-win_amd64.whl">paddlepaddle_gpu-2.6.0.dev0+cu118.ampere.llm-cp37-cp37m-win_amd64.whl</a> </td>
        <td> <a href="https://paddle-qa.bj.bcebos.com/paddle-pipeline/LLM-Windows-Gpu-Cuda11.8-Cudnn8.6-Trt8.5-Mkl-Avx-Vs2019/latest/paddlepaddle_gpu-2.6.0.dev0%2Bcu118.ampere.llm-cp38-cp38-win_amd64.whl">paddlepaddle_gpu-2.6.0.dev0+cu118.ampere.llm-cp38-cp38-win_amd64.whl</a> </td>
        <td> <a href="https://paddle-qa.bj.bcebos.com/paddle-pipeline/LLM-Windows-Gpu-Cuda11.8-Cudnn8.6-Trt8.5-Mkl-Avx-Vs2019/latest/paddlepaddle_gpu-2.6.0.dev0%2Bcu118.ampere.llm-cp39-cp39-win_amd64.whl">paddlepaddle_gpu-2.6.0.dev0+cu118.ampere.llm-cp39-cp39-win_amd64.whl</a> </td>
        <td> <a href="https://paddle-qa.bj.bcebos.com/paddle-pipeline/LLM-Windows-Gpu-Cuda11.8-Cudnn8.6-Trt8.5-Mkl-Avx-Vs2019/latest/paddlepaddle_gpu-2.6.0.dev0%2Bcu118.ampere.llm-cp310-cp310-win_amd64.whl">paddlepaddle_gpu-2.6.0.dev0+cu118.ampere.llm-cp310-cp310-win_amd64.whl</a> </td>
        <td> <a href="https://paddle-qa.bj.bcebos.com/paddle-pipeline/LLM-Windows-Gpu-Cuda11.8-Cudnn8.6-Trt8.5-Mkl-Avx-Vs2019/latest/paddlepaddle_gpu-2.6.0.dev0%2Bcu118.ampere.llm-cp311-cp311-win_amd64.whl">paddlepaddle_gpu-2.6.0.dev0+cu118.ampere.llm-cp311-cp311-win_amd64.whl</a> </td>
    </tr>
    <tr>
        <td> windows-cuda12.0-cudnn8.9-trt8.6-mkl-vs2019-avx-llm-volta </td>
        <td> <a href="https://paddle-qa.bj.bcebos.com/paddle-pipeline/LLM-Windows-Gpu-Cuda12.0-Cudnn8.9-Trt8.6-Mkl-Avx-Vs2019/latest/paddlepaddle_gpu-2.6.0.dev0%2Bcu120.volta.llm-cp37-cp37m-win_amd64.whl">paddlepaddle_gpu-2.6.0.dev0+cu120.volta.llm-cp37-cp37m-win_amd64.whl</a> </td>
        <td> <a href="https://paddle-qa.bj.bcebos.com/paddle-pipeline/LLM-Windows-Gpu-Cuda12.0-Cudnn8.9-Trt8.6-Mkl-Avx-Vs2019/latest/paddlepaddle_gpu-2.6.0.dev0%2Bcu120.volta.llm-cp38-cp38-win_amd64.whl">paddlepaddle_gpu-2.6.0.dev0+cu120.volta.llm-cp38-cp38-win_amd64.whl</a> </td>
        <td> <a href="https://paddle-qa.bj.bcebos.com/paddle-pipeline/LLM-Windows-Gpu-Cuda12.0-Cudnn8.9-Trt8.6-Mkl-Avx-Vs2019/latest/paddlepaddle_gpu-2.6.0.dev0%2Bcu120.volta.llm-cp39-cp39-win_amd64.whl">paddlepaddle_gpu-2.6.0.dev0+cu120.volta.llm-cp39-cp39-win_amd64.whl</a> </td>
        <td> <a href="https://paddle-qa.bj.bcebos.com/paddle-pipeline/LLM-Windows-Gpu-Cuda12.0-Cudnn8.9-Trt8.6-Mkl-Avx-Vs2019/latest/paddlepaddle_gpu-2.6.0.dev0%2Bcu120.volta.llm-cp310-cp310-win_amd64.whl">paddlepaddle_gpu-2.6.0.dev0+cu120.volta.llm-cp310-cp310-win_amd64.whl</a> </td>
        <td> <a href="https://paddle-qa.bj.bcebos.com/paddle-pipeline/LLM-Windows-Gpu-Cuda12.0-Cudnn8.9-Trt8.6-Mkl-Avx-Vs2019/latest/paddlepaddle_gpu-2.6.0.dev0%2Bcu120.volta.llm-cp311-cp311-win_amd64.whl">paddlepaddle_gpu-2.6.0.dev0+cu120.volta.llm-cp311-cp311-win_amd64.wh</a> </td>
    </tr>
    <tr>
        <td> windows-cuda12.0-cudnn8.9-trt8.6-mkl-vs2019-avx-llm-ampere </td>
        <td> <a href="https://paddle-qa.bj.bcebos.com/paddle-pipeline/LLM-Windows-Gpu-Cuda12.0-Cudnn8.9-Trt8.6-Mkl-Avx-Vs2019/latest/paddlepaddle_gpu-2.6.0.dev0%2Bcu120.ampere.llm-cp37-cp37m-win_amd64.whl">paddlepaddle_gpu-2.6.0.dev0+cu120.ampere.llm-cp37-cp37m-win_amd64.whl</a> </td>
        <td> <a href="https://paddle-qa.bj.bcebos.com/paddle-pipeline/LLM-Windows-Gpu-Cuda12.0-Cudnn8.9-Trt8.6-Mkl-Avx-Vs2019/latest/paddlepaddle_gpu-2.6.0.dev0%2Bcu120.ampere.llm-cp38-cp38-win_amd64.whl">paddlepaddle_gpu-2.6.0.dev0+cu120.ampere.llm-cp38-cp38-win_amd64.whl</a> </td>
        <td> <a href="https://paddle-qa.bj.bcebos.com/paddle-pipeline/LLM-Windows-Gpu-Cuda12.0-Cudnn8.9-Trt8.6-Mkl-Avx-Vs2019/latest/paddlepaddle_gpu-2.6.0.dev0%2Bcu120.ampere.llm-cp39-cp39-win_amd64.whl">paddlepaddle_gpu-2.6.0.dev0+cu120.ampere.llm-cp39-cp39-win_amd64.whl</a> </td>
        <td> <a href="https://paddle-qa.bj.bcebos.com/paddle-pipeline/LLM-Windows-Gpu-Cuda12.0-Cudnn8.9-Trt8.6-Mkl-Avx-Vs2019/latest/paddlepaddle_gpu-2.6.0.dev0%2Bcu120.ampere.llm-cp310-cp310-win_amd64.whl">paddlepaddle_gpu-2.6.0.dev0+cu120.ampere.llm-cp310-cp310-win_amd64.whl</a> </td>
        <td> <a href="https://paddle-qa.bj.bcebos.com/paddle-pipeline/LLM-Windows-Gpu-Cuda12.0-Cudnn8.9-Trt8.6-Mkl-Avx-Vs2019/latest/paddlepaddle_gpu-2.6.0.dev0%2Bcu120.ampere.llm-cp311-cp311-win_amd64.whl">paddlepaddle_gpu-2.6.0.dev0+cu120.ampere.llm-cp311-cp311-win_amd64.whl</a> </td>
    </tr>
   </tbody>
</table>
</p>

### Table instruction

- Vertical axis

cpu-mkl: Support CPU training and prediction, use Intel MKL math library

cuda10_cudnn7-mkl: Support GPU training and prediction, use Intel MKL math library


- Transverse axis

Generally, it is similar to "cp37-cp37m", in which:

37:python tag, refers to python3.7. Similarly, there are "36", "38", "39", etc

mu:refers to unicode version python, if it is m, refers to non Unicode version Python

- Installation package naming rules

Each installation package has a unique name. They are named according to the official rules of Python. The form is as follows:

{distribution}-{version}(-{build tag})?-{python tag}-{abi tag}-{platform tag}.whl

The build tag can be missing, and other parts cannot be missing

distribution: wheel name

version: Version, for example 0.14.0 (must be in numeric format)

python tag: similar to 'py36', 'py37', 'py38', 'py39', used to indicate the corresponding Python version

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
        <th> cp36-cp36m    </th>
        <th> cp37-cp37m    </th>
        <th> cp38-cp38    </th>
        <th> cp39-cp39    </th>
    </tr>
    </thead>
    <tbody>
    <tr>
        <td> linux-cpu-mkl </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/linux/linux-cpu-mkl-avx/paddlepaddle-0.0.0-cp36-cp36m-linux_x86_64.whl"> paddlepaddle-latest-cp36-cp36m-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/linux/linux-cpu-mkl-avx/paddlepaddle-0.0.0-cp37-cp37m-linux_x86_64.whl"> paddlepaddle-latest-cp37-cp37m-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/linux/linux-cpu-mkl-avx/paddlepaddle-0.0.0-cp38-cp38-linux_x86_64.whl"> paddlepaddle-latest-cp38-cp38-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/linux/linux-cpu-mkl-avx/paddlepaddle-0.0.0-cp39-cp39-linux_x86_64.whl"> paddlepaddle-latest-cp39-cp39-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/linux/linux-cpu-mkl-avx/paddlepaddle-0.0.0-cp310-cp310-linux_x86_64.whl"> paddlepaddle-latest-cp310-cp310-linux_x86_64.whl</a></td>
    </tr>
    <tr>
        <td> linux-cpu-openblas </td>
        <td> - </td>
        <td> - </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/linux/linux-cpu-openblas-avx/paddlepaddle-0.0.0-cp38-cp38-linux_x86_64.whl"> paddlepaddle-latest-cp38-cp38-linux_x86_64.whl</a></td>
        <td> - </td>
        <td> - </td>
    </tr>
    <tr>
        <td> linux-cuda10.1-cudnn7-mkl-gcc5.4 </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/linux/linux-gpu-cuda10.1-cudnn7-mkl-gcc5.4-avx/paddlepaddle_gpu-0.0.0.post101-cp36-cp36m-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp36-cp36m-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/linux/linux-gpu-cuda10.1-cudnn7-mkl-gcc5.4-avx/paddlepaddle_gpu-0.0.0.post101-cp37-cp37m-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp37-cp37m-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/linux/linux-gpu-cuda10.1-cudnn7-mkl-gcc5.4-avx/paddlepaddle_gpu-0.0.0.post101-cp38-cp38-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp38-cp38-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/linux/linux-gpu-cuda10.1-cudnn7-mkl-gcc5.4-avx/paddlepaddle_gpu-0.0.0.post101-cp39-cp39-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp39-cp39-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/linux/linux-gpu-cuda10.1-cudnn7-mkl-gcc5.4-avx/paddlepaddle_gpu-0.0.0.post101-cp310-cp310-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp310-cp310-linux_x86_64.whl</a></td>
    </tr>
    <tr>
        <td> linux-cuda10.2-cudnn7-mkl </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/linux/linux-gpu-cuda10.2-cudnn7-mkl-gcc8.2-avx/paddlepaddle_gpu-0.0.0.post102-cp36-cp36m-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp36-cp36m-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/linux/linux-gpu-cuda10.2-cudnn7-mkl-gcc8.2-avx/paddlepaddle_gpu-0.0.0.post102-cp37-cp37m-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp37-cp37m-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/linux/linux-gpu-cuda10.2-cudnn7-mkl-gcc8.2-avx/paddlepaddle_gpu-0.0.0.post102-cp38-cp38-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp38-cp38-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/linux/linux-gpu-cuda10.2-cudnn7-mkl-gcc8.2-avx/paddlepaddle_gpu-0.0.0.post102-cp39-cp39-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp39-cp39-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/linux/linux-gpu-cuda10.2-cudnn7-mkl-gcc8.2-avx/paddlepaddle_gpu-0.0.0.post102-cp310-cp310-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp310-cp310-linux_x86_64.whl</a></td>
    </tr>
    <tr>
        <td> linux-cuda11.0-cudnn8.0-mkl </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/linux/linux-gpu-cuda11.0-cudnn8-mkl-gcc8.2-avx/paddlepaddle_gpu-0.0.0.post110-cp36-cp36m-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp36-cp36m-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/linux/linux-gpu-cuda11.0-cudnn8-mkl-gcc8.2-avx/paddlepaddle_gpu-0.0.0.post110-cp37-cp37m-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp37-cp37m-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/linux/linux-gpu-cuda11.0-cudnn8-mkl-gcc8.2-avx/paddlepaddle_gpu-0.0.0.post110-cp38-cp38-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp38-cp38-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/linux/linux-gpu-cuda11.0-cudnn8-mkl-gcc8.2-avx/paddlepaddle_gpu-0.0.0.post110-cp39-cp39-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp39-cp39-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/linux/linux-gpu-cuda11.0-cudnn8-mkl-gcc8.2-avx/paddlepaddle_gpu-0.0.0.post110-cp310-cp310-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp310-cp310-linux_x86_64.whl</a></td>
    </tr>
    <tr>
        <td> linux-cuda11.1-cudnn8.1-mkl </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/linux/linux-gpu-cuda11.1-cudnn8.1-mkl-gcc8.2-avx/paddlepaddle_gpu-0.0.0.post111-cp36-cp36m-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp36-cp36m-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/linux/linux-gpu-cuda11.1-cudnn8.1-mkl-gcc8.2-avx/paddlepaddle_gpu-0.0.0.post111-cp37-cp37m-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp37-cp37m-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/linux/linux-gpu-cuda11.1-cudnn8.1-mkl-gcc8.2-avx/paddlepaddle_gpu-0.0.0.post111-cp38-cp38-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp38-cp38-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/linux/linux-gpu-cuda11.1-cudnn8.1-mkl-gcc8.2-avx/paddlepaddle_gpu-0.0.0.post111-cp39-cp39-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp39-cp39-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/linux/linux-gpu-cuda11.1-cudnn8.1-mkl-gcc8.2-avx/paddlepaddle_gpu-0.0.0.post111-cp310-cp310-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp310-cp310-linux_x86_64.whl</a></td>
    </tr>
    <tr>
        <td> linux-cuda11.2-cudnn8.1-mkl </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/linux/linux-gpu-cuda11.2-cudnn8-mkl-gcc8.2-avx/paddlepaddle_gpu-0.0.0.post112-cp36-cp36m-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp36-cp36m-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/linux/linux-gpu-cuda11.2-cudnn8-mkl-gcc8.2-avx/paddlepaddle_gpu-0.0.0.post112-cp37-cp37m-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp37-cp37m-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/linux/linux-gpu-cuda11.2-cudnn8-mkl-gcc8.2-avx/paddlepaddle_gpu-0.0.0.post112-cp38-cp38-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp38-cp38-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/linux/linux-gpu-cuda11.2-cudnn8-mkl-gcc8.2-avx/paddlepaddle_gpu-0.0.0.post112-cp39-cp39-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp39-cp39-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/linux/linux-gpu-cuda11.2-cudnn8-mkl-gcc8.2-avx/paddlepaddle_gpu-0.0.0.post112-cp310-cp310-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp310-cp310-linux_x86_64.whl</a></td>
    </tr>
    <tr>
        <td> linux-cuda11.6-cudnn8.4-mkl </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/linux/linux-gpu-cuda11.6-cudnn8.4.0-mkl-gcc8.2-avx/paddlepaddle_gpu-0.0.0.post116-cp36-cp36m-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp36-cp36m-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/linux/linux-gpu-cuda11.6-cudnn8.4.0-mkl-gcc8.2-avx/paddlepaddle_gpu-0.0.0.post116-cp37-cp37m-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp37-cp37m-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/linux/linux-gpu-cuda11.6-cudnn8.4.0-mkl-gcc8.2-avx/paddlepaddle_gpu-0.0.0.post116-cp38-cp38-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp38-cp38-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/linux/linux-gpu-cuda11.6-cudnn8.4.0-mkl-gcc8.2-avx/paddlepaddle_gpu-0.0.0.post116-cp39-cp39-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp39-cp39-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/linux/linux-gpu-cuda11.6-cudnn8.4.0-mkl-gcc8.2-avx/paddlepaddle_gpu-0.0.0.post116-cp310-cp310-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp310-cp310-linux_x86_64.whl</a></td>
    </tr>
    <tr>
        <td> linuxcuda11.7-cudnn8.4-mkl </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/linux/linux-gpu-cuda11.7-cudnn8.4.1-mkl-gcc8.2-avx/paddlepaddle_gpu-0.0.0.post117-cp36-cp36m-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp36-cp36m-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/linux/linux-gpu-cuda11.7-cudnn8.4.1-mkl-gcc8.2-avx/paddlepaddle_gpu-0.0.0.post117-cp37-cp37m-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp37-cp37m-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/linux/linux-gpu-cuda11.7-cudnn8.4.1-mkl-gcc8.2-avx/paddlepaddle_gpu-0.0.0.post117-cp38-cp38-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp38-cp38-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/linux/linux-gpu-cuda11.7-cudnn8.4.1-mkl-gcc8.2-avx/paddlepaddle_gpu-0.0.0.post117-cp39-cp39-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp39-cp39-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/linux/linux-gpu-cuda11.7-cudnn8.4.1-mkl-gcc8.2-avx/paddlepaddle_gpu-0.0.0.post117-cp310-cp310-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp310-cp310-linux_x86_64.whl</a></td>
    </tr>
    <tr>
        <td> mac-cpu </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/macos/macos-cpu-openblas/paddlepaddle-0.0.0-cp36-cp36m-macosx_10_6_intel.whl"> paddlepaddle-cp36-cp36m-macosx_10_6_intel.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/macos/macos-cpu-openblas/paddlepaddle-0.0.0-cp37-cp37m-macosx_10_14_intel.whl"> paddlepaddle-cp37-cp37m-macosx_10_6_intel.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/macos/macos-cpu-openblas/paddlepaddle-0.0.0-cp38-cp38-macosx_10_14_x86_64.whl"> paddlepaddle-cp38-cp38-macosx_10_14_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/macos/macos-cpu-openblas/paddlepaddle-0.0.0-cp39-cp39-macosx_10_14_x86_64.whl"> paddlepaddle-cp39-cp39-macosx_10_14_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/macos/macos-cpu-openblas/paddlepaddle-0.0.0-cp310-cp310-macosx_10_14_universal2.whl"> paddlepaddle-cp310-cp310-macosx_10_14_universal2.whl</a></td>
    </tr>
    <tr>
        <td> win-cpu-mkl-avx </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/windows/windows-cpu-avx-mkl-vs2017/paddlepaddle-0.0.0-cp36-cp36m-win_amd64.whl"> paddlepaddle-latest-cp36-cp36m-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/windows/windows-cpu-avx-mkl-vs2017/paddlepaddle-0.0.0-cp37-cp37m-win_amd64.whl"> paddlepaddle-latest-cp37-cp37m-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/windows/windows-cpu-avx-mkl-vs2017/paddlepaddle-0.0.0-cp38-cp38-win_amd64.whl"> paddlepaddle-latest-cp38-cp38-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/windows/windows-cpu-avx-mkl-vs2017/paddlepaddle-0.0.0-cp39-cp39-win_amd64.whl"> paddlepaddle-latest-cp39-cp39-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/windows/windows-cpu-avx-mkl-vs2017/paddlepaddle-0.0.0-cp310-cp310-win_amd64.whl"> paddlepaddle-latest-cp310-cp310-win_amd64.whl</a></td>
    </tr>
    <tr>
        <td> win-cpu-mkl-noavx </td>
        <td> - </td>
        <td> - </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/windows/windows-cpu-noavx-mkl-vs2017/paddlepaddle-0.0.0-cp38-cp38-win_amd64.whl"> paddlepaddle-latest-cp38-cp38-win_amd64.whl</a></td>
        <td> - </td>
        <td> - </td>
    </tr>
    <tr>
        <td> win-cpu-openblas-avx </td>
        <td> - </td>
        <td> - </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/windows/windows-cpu-avx-openblas-vs2017/paddlepaddle-0.0.0-cp38-cp38-win_amd64.whl"> paddlepaddle-latest-cp38-cp38-win_amd64.whl</a></td>
        <td> - </td>
        <td> - </td>
    </tr>
    <tr>
        <td> win-cpu-openblas-noavx </td>
        <td> - </td>
        <td> - </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/windows/windows-cpu-noavx-openblas-vs2017/paddlepaddle-0.0.0-cp38-cp38-win_amd64.whl"> paddlepaddle-latest-cp38-cp38-win_amd64.whl</a></td>
        <td> - </td>
        <td> - </td>
    </tr>
    <tr>
        <td> win-cuda10.2-cudnn7-mkl-vs2017-avx </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/windows/windows-gpu-cuda10.2-cudnn7.6.5-mkl-avx-vs2017/paddlepaddle_gpu-0.0.0.post102-cp36-cp36m-win_amd64.whl"> paddlepaddle_gpu-latest-cp36-cp36m-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/windows/windows-gpu-cuda10.2-cudnn7.6.5-mkl-avx-vs2017/paddlepaddle_gpu-0.0.0.post102-cp37-cp37m-win_amd64.whl"> paddlepaddle_gpu-latest-cp37-cp37m-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/windows/windows-gpu-cuda10.2-cudnn7.6.5-mkl-avx-vs2017/paddlepaddle_gpu-0.0.0.post102-cp38-cp38-win_amd64.whl"> paddlepaddle_gpu-latest-cp38-cp38-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/windows/windows-gpu-cuda10.2-cudnn7.6.5-mkl-avx-vs2017/paddlepaddle_gpu-0.0.0.post102-cp39-cp39-win_amd64.whl"> paddlepaddle_gpu-latest-cp39-cp39-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/windows/windows-gpu-cuda10.2-cudnn7.6.5-mkl-avx-vs2017/paddlepaddle_gpu-0.0.0.post102-cp310-cp310-win_amd64.whl"> paddlepaddle_gpu-latest-cp310-cp310-win_amd64.whl</a></td>
    </tr>
    <tr>
        <td> win-cuda10.2-cudnn7-mkl-vs2017-noavx </td>
        <td> - </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/windows/windows-gpu-cuda10.2-cudnn7.6.5-mkl-noavx-vs2017/paddlepaddle_gpu-0.0.0.post102-cp37-cp37m-win_amd64.whl"> paddlepaddle_gpu-latest-cp37-cp37m-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/windows/windows-gpu-cuda10.2-cudnn7.6.5-mkl-noavx-vs2017/paddlepaddle_gpu-0.0.0.post102-cp38-cp38-win_amd64.whl"> paddlepaddle_gpu-latest-cp38-cp38-win_amd64.whl</a></td>
        <td> - </td>
        <td> - </td>
    </tr>
    <tr>
        <td> win-cuda11.0-cudnn8.0-mkl-vs2017-avx </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/windows/windows-gpu-cuda11.0-cudnn8.0.2-mkl-avx-vs2017/paddlepaddle_gpu-0.0.0.post110-cp36-cp36m-win_amd64.whl"> paddlepaddle_gpu-latest-cp36-cp36m-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/windows/windows-gpu-cuda11.0-cudnn8.0.2-mkl-avx-vs2017/paddlepaddle_gpu-0.0.0.post110-cp37-cp37m-win_amd64.whl"> paddlepaddle_gpu-latest-cp37-cp37m-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/windows/windows-gpu-cuda11.0-cudnn8.0.2-mkl-avx-vs2017/paddlepaddle_gpu-0.0.0.post110-cp38-cp38-win_amd64.whl"> paddlepaddle_gpu-latest-cp38-cp38-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/windows/windows-gpu-cuda11.0-cudnn8.0.2-mkl-avx-vs2017/paddlepaddle_gpu-0.0.0.post110-cp39-cp39-win_amd64.whl"> paddlepaddle_gpu-latest-cp39-cp39-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/windows/windows-gpu-cuda11.0-cudnn8.0.2-mkl-avx-vs2017/paddlepaddle_gpu-0.0.0.post110-cp310-cp310-win_amd64.whl"> paddlepaddle_gpu-latest-cp310-cp310-win_amd64.whl</a></td>
    </tr>
    <tr>
        <td> win-cuda11.1-cudnn8.1-mkl-vs2017-avx </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/windows/windows-gpu-cuda11.1-cudnn8.1.1-mkl-avx-vs2017/paddlepaddle_gpu-0.0.0.post111-cp36-cp36m-win_amd64.whl"> paddlepaddle_gpu-latest-cp36-cp36m-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/windows/windows-gpu-cuda11.1-cudnn8.1.1-mkl-avx-vs2017/paddlepaddle_gpu-0.0.0.post111-cp37-cp37m-win_amd64.whl"> paddlepaddle_gpu-latest-cp37-cp37m-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/windows/windows-gpu-cuda11.1-cudnn8.1.1-mkl-avx-vs2017/paddlepaddle_gpu-0.0.0.post111-cp38-cp38-win_amd64.whl"> paddlepaddle_gpu-latest-cp38-cp38-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/windows/windows-gpu-cuda11.1-cudnn8.1.1-mkl-avx-vs2017/paddlepaddle_gpu-0.0.0.post111-cp39-cp39-win_amd64.whl"> paddlepaddle_gpu-latest-cp39-cp39-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/windows/windows-gpu-cuda11.1-cudnn8.1.1-mkl-avx-vs2017/paddlepaddle_gpu-0.0.0.post111-cp310-cp310-win_amd64.whl"> paddlepaddle_gpu-latest-cp310-cp310-win_amd64.whl</a></td>
    </tr>
    <tr>
        <td> win-cuda11.2-cudnn8.2-mkl-vs2019-avx </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/windows/windows-gpu-cuda11.2-cudnn8.2.1-mkl-avx-vs2019/paddlepaddle_gpu-0.0.0.post112-cp36-cp36m-win_amd64.whl"> paddlepaddle_gpu-latest-cp36-cp36m-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/windows/windows-gpu-cuda11.2-cudnn8.2.1-mkl-avx-vs2019/paddlepaddle_gpu-0.0.0.post112-cp37-cp37m-win_amd64.whl"> paddlepaddle_gpu-latest-cp37-cp37m-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/windows/windows-gpu-cuda11.2-cudnn8.2.1-mkl-avx-vs2019/paddlepaddle_gpu-0.0.0.post112-cp38-cp38-win_amd64.whl"> paddlepaddle_gpu-latest-cp38-cp38-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/windows/windows-gpu-cuda11.2-cudnn8.2.1-mkl-avx-vs2019/paddlepaddle_gpu-0.0.0.post112-cp39-cp39-win_amd64.whl"> paddlepaddle_gpu-latest-cp39-cp39-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/windows/windows-gpu-cuda11.2-cudnn8.2.1-mkl-avx-vs2019/paddlepaddle_gpu-0.0.0.post112-cp310-cp310-win_amd64.whl"> paddlepaddle_gpu-latest-cp310-cp310-win_amd64.whl</a></td>
    </tr>
    <tr>
        <td> win-cuda11.6-cudnn8.4-mkl-vs2019-avx </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/windows/windows-gpu-cuda11.6-cudnn8.4.0-mkl-avx-vs2019/paddlepaddle_gpu-0.0.0.post116-cp36-cp36m-win_amd64.whl"> paddlepaddle_gpu-latest-cp36-cp36m-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/windows/windows-gpu-cuda11.6-cudnn8.4.0-mkl-avx-vs2019/paddlepaddle_gpu-0.0.0.post116-cp37-cp37m-win_amd64.whl"> paddlepaddle_gpu-latest-cp37-cp37m-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/windows/windows-gpu-cuda11.6-cudnn8.4.0-mkl-avx-vs2019/paddlepaddle_gpu-0.0.0.post116-cp38-cp38-win_amd64.whl"> paddlepaddle_gpu-latest-cp38-cp38-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/windows/windows-gpu-cuda11.6-cudnn8.4.0-mkl-avx-vs2019/paddlepaddle_gpu-0.0.0.post116-cp39-cp39-win_amd64.whl"> paddlepaddle_gpu-latest-cp39-cp39-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/windows/windows-gpu-cuda11.6-cudnn8.4.0-mkl-avx-vs2019/paddlepaddle_gpu-0.0.0.post116-cp310-cp310-win_amd64.whl"> paddlepaddle_gpu-latest-cp310-cp310-win_amd64.whl</a></td>
    </tr>
    <tr>
        <td> win-cuda11.7-cudnn8.4-mkl-vs2019-avx </td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/windows/windows-gpu-cuda11.7-cudnn8.4.1-mkl-avx-vs2019/paddlepaddle_gpu-0.0.0.post117-cp36-cp36m-win_amd64.whl"> paddlepaddle_gpu-latest-cp36-cp36m-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/windows/windows-gpu-cuda11.7-cudnn8.4.1-mkl-avx-vs2019/paddlepaddle_gpu-0.0.0.post117-cp37-cp37m-win_amd64.whl"> paddlepaddle_gpu-latest-cp37-cp37m-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/windows/windows-gpu-cuda11.7-cudnn8.4.1-mkl-avx-vs2019/paddlepaddle_gpu-0.0.0.post117-cp38-cp38-win_amd64.whl"> paddlepaddle_gpu-latest-cp38-cp38-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/windows/windows-gpu-cuda11.7-cudnn8.4.1-mkl-avx-vs2019/paddlepaddle_gpu-0.0.0.post117-cp39-cp39-win_amd64.whl"> paddlepaddle_gpu-latest-cp39-cp39-win_amd64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/develop/windows/windows-gpu-cuda11.7-cudnn8.4.1-mkl-avx-vs2019/paddlepaddle_gpu-0.0.0.post117-cp310-cp310-win_amd64.whl"> paddlepaddle_gpu-latest-cp310-cp310-win_amd64.whl</a></td>
    </tr>
   </tbody>
</table>
</p>



<!--TODO this part should be in a new webpage-->

</br></br>


## Execute the PaddlePaddle training program in Docker


Suppose you have written a PaddlePaddle program in the current directory (such as /home/work): `train.py` ( refer to [PaddlePaddleBook](https://github.com/PaddlePaddle/book/blob/develop/01.fit_a_line/README.cn.md) to write), you can start the training with the following command:

```
cd /home/work
```
```
docker run -it -v $PWD:/work registry.baidubce.com/paddlepaddle/paddle /work/train.py
```


In the above commands, the `-it` parameter indicates that the container has been run interactively; `-v $PWD:/work` specifies that the current path (the absolute path where the PWD variable in Linux will expand to the current path) is mounted to the `:/work` directory inside the container: `registry.baidubce.com/paddlepaddle/paddle` specifies the container to be used; finally `/work/train.py` is the command executed inside the container, ie. the training program.

Of course, you can also enter into the Docker container and execute or debug your code interactively:

```
docker run -it -v $PWD:/work registry.baidubce.com/paddlepaddle/paddle /bin/bash
```
```
cd /work
```
```
python train.py
```


**Note: In order to reduce the size, vim is not installed in PaddlePaddle Docker image by default. You can edit the code in the container after executing ** `apt-get install -y vim` **(which installs vim for you) in the container.**

</br></br>

## Start PaddlePaddle Book tutorial with Docker


Use Docker to quickly launch a local Jupyter Notebook containing the PaddlePaddle official Book tutorial, which can be viewed on the web. PaddlePaddle Book is an interactive Jupyter Notebook for users and developers. If you want to learn more about deep learning, PaddlePaddle Book is definitely your best choice. You can read tutorials or create and share interactive documents with code, formulas, charts, and text.

We provide a Docker image that can run the PaddlePaddle Book directly, running directly:

```
docker run -p 8888:8888 registry.baidubce.com/paddlepaddle/book
```

Domestic users can use the following image source to speed up access:

```
docker run -p 8888:8888 registry.baidubce.com/paddlepaddle/book
```

Then enter the following URL in your browser:

```
http://localhost:8888/
```



</br></br>
## Perform GPU training using Docker


In order to ensure that the GPU driver works properly in the image, we recommend using [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) to run the image. Don't forget to install the latest GPU drivers on your physical machine in advance.

```
Nvidia-docker run -it -v $PWD:/work registry.baidubce.com/paddlepaddle/paddle:latest-gpu /bin/bash
```

**Note: If you don't have nvidia-docker installed, you can try the following to mount the CUDA library and Linux devices into the Docker container:**

```
export CUDA_SO="$(\ls /usr/lib64/libcuda* | xargs -I{} echo '-v {}:{}') \
$(\ls /usr/lib64/libnvidia* | xargs -I{} echo '-v {}:{}')"
export DEVICES=$(\ls /dev/nvidia* | xargs -I{} echo '--device {}:{}')
docker run ${CUDA_SO} \
${DEVICES} -it registry.baidubce.com/paddlepaddle/paddle:latest-gpu
```
