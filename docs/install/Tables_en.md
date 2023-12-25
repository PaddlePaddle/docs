<a name="third_party"></a>
# Appendix


<a name="gpu-install"></a>

## **Nvidia GPU architecture and installation mode supported by PaddlePaddle**
<p align="center">
<table>
    <thead>
    <tr>
        <th> GPU </th>
        <th> Compute Capability </th>
        <th> Corresponding GPU hardware model </th>
        <th> Please download the following CUDA version of PaddlePaddle installation package </th>
    </tr>
    </thead>
    <tbody>
    <tr>
        <td> Pascal </td>
        <td> sm_60 </td>
        <td> Quadro GP100, Tesla P100, DGX-1 </td>
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
        <td> CUDA11、CUDA11.2 (Recommend) </td>
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
        <td> 3.18, 3.19(Recommend) </td>
        <td>  </td>
        <td>  </td>
    </tr>
    <tr>
        <td> GCC </td>
        <td> 8.2 / 12.2 </td>
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
        <td> 3.8+.x </td>
        <td> depends on libpython3.8+.so </td>
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
        <td> paddlepaddle==[version code] such as paddlepaddle==2.6.0 </td>
        <td> Only support the corresponding version of the CPU PaddlePaddle, please refer to <a href=https://pypi.org/project/paddlepaddle/#history>Pypi</a> for the specific version. </td>
    </tr>
    <tr>
        <td> paddlepaddle-gpu==[version code], such as paddlepaddle-gpu==2.6.0 </td>
        <td> The default installation supports the PaddlePaddle installation package corresponding to [version number] of CUDA 11.2 and cuDNN 8 </td>
    </tr>
   </tbody>
</table>
</p>

You can find various distributions of PaddlePaddle-gpu in [the Release History](https://pypi.org/project/paddlepaddle-gpu/#history).
> 'postxx' corresponds to CUDA and cuDNN versions, and the number before 'postxx' represents the version of Paddle

Please note that: in the commands, <code> paddlepaddle-gpu==2.6.0 </code> will install the installation package of PaddlePaddle that supports CUDA 11.2 and cuDNN 8 by default under Windows environment.


<a name="ciwhls-release"></a>
</br></br>

## **Multi-version whl package list - Release**


<p align="center">
<table>
    <thead>
    <tr>
        <th> Release Instruction </th>
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
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.6.0/linux/linux-gpu-cuda11.6-cudnn8.4.0-mkl-gcc8.2-avx/paddlepaddle_gpu-2.6.0.post116-cp312-cp312-linux_x86_64.whl">
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
        paddlepaddle_gpu-2.6.0.post117-cp311-cp311-linux_x86_64.whl</a></td>
        <td> <a href="https://paddle-wheel.bj.bcebos.com/2.6.0/linux/linux-gpu-cuda11.7-cudnn8.4.1-mkl-gcc8.2-avx/paddlepaddle_gpu-2.6.0.post117-cp312-cp312-linux_x86_64.whl">
        paddlepaddle_gpu-2.6.0.post117-cp312-cp312-linux_x86_64.whl</a></td>
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


### Table instruction

- Vertical axis

cpu-mkl: Support CPU training and prediction, use Intel MKL math library

cuda10_cudnn7-mkl: Support GPU training and prediction, use Intel MKL math library


- Transverse axis

Generally, it is similar to "cp310-cp310", in which:

310:python tag, refers to python3.10. Similarly, there are "38", "39", "310", "311", "312", etc

mu:refers to unicode version python, if it is m, refers to non Unicode version Python

- Installation package naming rules

Each installation package has a unique name. They are named according to the official rules of Python. The form is as follows:

{distribution}-{version}(-{build tag})?-{python tag}-{abi tag}-{platform tag}.whl

The build tag can be missing, and other parts cannot be missing

distribution: wheel name

version: Version, for example 0.14.0 (must be in numeric format)

python tag: similar to 'py38', 'py39', 'py310', 'py311', 'py312', used to indicate the corresponding Python version

abi tag: similar to 'cp33m', 'abi3', 'none'

platform tag: similar to 'linux_x86_64', 'any'


<a name="ciwhls"></a>
</br></br>
## **Multi-version whl package list - dev**
<p align="center">
<table>
    <thead>
    <tr>
        <th> Release Instruction </th>
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
