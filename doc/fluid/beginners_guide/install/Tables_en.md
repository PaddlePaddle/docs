***
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
		<td> 3.4 </td>
		<td>  </td>
		<td>  </td>
	</tr>
	<tr>
		<td> GCC </td>
		<td> 4.8 / 5.4 </td>
		<td>  recommends using devtools2 for CentOS </td>
		<td>  </td>
	</tr>
		<tr>
		<td> Python </td>
		<td> 2.7.x. </td>
		<td> depends on libpython2.7.so </td>
		<td> <code> apt install python-dev </code> or <code> yum install python-devel </code></td>
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
		<td>  </td>
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
		<td> <code> pip install numpy==1.14.0 </code> </td>
	</tr>
	<tr>
		<td> protobuf </td>
		<td> 3.1.0 </td>
		<td>  </td>
		<td> <code> pip install protobuf==3.1.0 </code> </td>
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
	</tbody>
</table>
</p>


***
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
		<td> WITH_C_API </td>
		<td> Whether to compile CAPI </td>
		<td>  OFF </td>
	</tr>
		<tr>
		<td> WITH_DOUBLE </td>
		<td> Whether to use double precision floating point numeber </td>
		<td> OFF </td>
	</tr>
	<tr>
		<td> WITH_DSO </td>
		<td> whether to load CUDA dynamic libraries dynamically at runtime, instead of statically loading CUDA dynamic libraries. </td>
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
		<td> WITH_STYLE_CHECK </td>
		<td> Whether to perform code style checking at compile time </td>
		<td> ON </td>
	</tr>
	<tr>
		<td> WITH_TESTING </td>
		<td>  Whether to turn on unit test </td>
		<td> OFF </td>
	</tr>
	<tr>
		<td> WITH_DOC </td>
		<td> Whether to compile Chinese and English documents </td>
		<td> OFF </td>
	</tr>
	<tr>
		<td> WITH_SWIG_PY </td>
		<td> Whether to compile PYTHON's SWIG interface, which can be used for predicting and customizing training </td>
		<td> Auto </td>
	<tr>
		<td> WITH_GOLANG </td>
		<td> Whether to compile the fault-tolerant parameter server of the go language </td>
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
		<td> WITH_RDMA </td>
		<td> Whether to compile the relevant parts that supports RDMA </td>
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
		<td> DWITH_ANAKIN </td>
		<td> Whether to Compile ANAKIN </td>
		<td> OFF </td>
	</tr>
   </tbody>
</table>
</p>


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


***
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
		<td> paddlepaddle==[version code] such as paddlepaddle==1.5.0 </td>
		<td> Only support the corresponding version of the CPU PaddlePaddle, please refer to <a href=https://pypi.org/project/paddlepaddle/#history>Pypi</a> for the specific version. </td>
	</tr>
	<tr>
		<td> paddlepaddle-gpu==1.5.0 </td>
		<td>  Using version 1.5.0 compiled with CUDA 9.0 and cuDNN 7 </td>
	</tr>
	<tr>
		<td> paddlepaddle-gpu==1.5.0.post87 </td>
		<td> Using version 1.5.0 compiled with CUDA 8.0 and cuDNN 7 </td>
	</tr>	
   </tbody>
</table>
</p>



You can find various distributions of PaddlePaddle-gpu in [the Release History](https://pypi.org/project/paddlepaddle-gpu/#history).

Please note that: paddlepaddle-gpu in windows, will download package compiled with CUDA 8.0 and cuDNN 7

***
<a name="dockers"></a>
</br></br>
## Installation Mirrors and Introduction

<p align="center">
<table>
	<thead>
	<tr>
		<th> Version Number </th>
		<th> Release Description </th>
	</tr>
	</thead>
	<tbody>
	<tr>
		<td> hub.baidubce.com/paddlepaddle/paddle:latest </td>
		<td> The latest pre-installed image of the PaddlePaddle CPU version </td>
	</tr>
	<tr>
		<td> hub.baidubce.com/paddlepaddle/paddle:latest-dev </td>
		<td> The latest PaddlePaddle development environment </td>
	</tr>
		<tr>
		<td> hub.baidubce.com/paddlepaddle/paddle:[Version] </td>
		<td>  Replace version with a specific version, preinstalled PaddlePaddle image in historical version </td>
	</tr>
	<tr>
		<td> hub.baidubce.com/paddlepaddle/paddle:latest-gpu </td>
		<td> The latest pre-installed image of the PaddlePaddle GPU version </td>
	</tr>
   </tbody>
</table>
</p>



You can find the docker image for each release of PaddlePaddle in the [DockerHub](https://hub.docker.com/r/paddlepaddle/paddle/tags/).

***
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
		<th> cp35-cp35m	</th>
		<th> cp36-cp36m	</th>
		<th> cp37-cp37m	</th>
	</tr>
	</thead>
	<tbody>
	<tr>
		<td> cpu-mkl </td>
		<td> <a href="https://paddle-wheel.bj.bcebos.com/1.5.0-cpu-mkl/paddlepaddle-1.5.0-cp27-cp27mu-linux_x86_64.whl">
		paddlepaddle-1.5.0-cp27-cp27mu-linux_x86_64.whl</a></td>
		<td> <a href="https://paddle-wheel.bj.bcebos.com/1.5.0-cpu-mkl/paddlepaddle-1.5.0-cp27-cp27m-linux_x86_64.whl">
		paddlepaddle-1.5.0-cp27-cp27m-linux_x86_64.whl</a></td>
		<td> <a href="https://paddle-wheel.bj.bcebos.com/1.5.0-cpu-mkl/paddlepaddle-1.5.0-cp35-cp35m-linux_x86_64.whl">
		paddlepaddle-1.5.0-cp35-cp35m-linux_x86_64.whl</a></td>
		<td> <a href="https://paddle-wheel.bj.bcebos.com/1.5.0-cpu-mkl/paddlepaddle-1.5.0-cp36-cp36m-linux_x86_64.whl">
		paddlepaddle-1.5.0-cp36-cp36m-linux_x86_64.whl</a></td>
		<td> <a href="https://paddle-wheel.bj.bcebos.com/1.5.0-cpu-mkl/paddlepaddle-1.5.0-cp37-cp37m-linux_x86_64.whl">
		paddlepaddle-1.5.0-cp37-cp37m-linux_x86_64.whl</a></td>
	</tr>	
	<tr>
		<td> cpu-openblas </td>
		<td> <a href="https://paddle-wheel.bj.bcebos.com/1.5.0-cpu-openblas/paddlepaddle-1.5.0-cp27-cp27mu-linux_x86_64.whl">
		paddlepaddle-1.5.0-cp27-cp27mu-linux_x86_64.whl</a></td>
		<td> <a href="https://paddle-wheel.bj.bcebos.com/1.5.0-cpu-openblas/paddlepaddle-1.5.0-cp27-cp27m-linux_x86_64.whl"> paddlepaddle-1.5.0-cp27-cp27m-linux_x86_64.whl</a></td>
		<td> <a href="https://paddle-wheel.bj.bcebos.com/1.5.0-cpu-openblas/paddlepaddle-1.5.0-cp35-cp35m-linux_x86_64.whl">
		paddlepaddle-1.5.0-cp35-cp35m-linux_x86_64.whl</a></td>
		<td> <a href="https://paddle-wheel.bj.bcebos.com/1.5.0-cpu-openblas/paddlepaddle-1.5.0-cp36-cp36m-linux_x86_64.whl">
		paddlepaddle-1.5.0-cp36-cp36m-linux_x86_64.whl</a></td>
		<td> <a href="https://paddle-wheel.bj.bcebos.com/1.5.0-cpu-openblas/paddlepaddle-1.5.0-cp37-cp37m-linux_x86_64.whl">
		paddlepaddle-1.5.0-cp37-cp37m-linux_x86_64.whl</a></td>
	</tr>
	<tr>
		<td> cuda8-cudnn7-openblas </td>
		<td> <a href="https://paddle-wheel.bj.bcebos.com/1.5.0-gpu-cuda8-cudnn7-openblas/paddlepaddle_gpu-1.5.0-cp27-cp27mu-linux_x86_64.whl"> paddlepaddle_gpu-1.5.0-cp27-cp27mu-linux_x86_64.whl</a></td>
		<td> <a href="https://paddle-wheel.bj.bcebos.com/1.5.0-gpu-cuda8-cudnn7-openblas/paddlepaddle_gpu-1.5.0-cp27-cp27m-linux_x86_64.whl"> paddlepaddle_gpu-1.5.0-cp27-cp27m-linux_x86_64.whl</a></td>
		<td> <a href="https://paddle-wheel.bj.bcebos.com/1.5.0-gpu-cuda8-cudnn7-openblas/paddlepaddle_gpu-1.5.0-cp35-cp35m-linux_x86_64.whl"> paddlepaddle_gpu-1.5.0-cp35-cp35m-linux_x86_64.whl</a></td>
		<td> <a href="https://paddle-wheel.bj.bcebos.com/1.5.0-gpu-cuda8-cudnn7-openblas/paddlepaddle_gpu-1.5.0-cp36-cp36m-linux_x86_64.whl"> paddlepaddle_gpu-1.5.0-cp36-cp36m-linux_x86_64.whl</a></td>
		<td> <a href="https://paddle-wheel.bj.bcebos.com/1.5.0-gpu-cuda8-cudnn7-openblas/paddlepaddle_gpu-1.5.0-cp37-cp37m-linux_x86_64.whl"> paddlepaddle_gpu-1.5.0-cp37-cp37m-linux_x86_64.whl</a></td>
	</tr>
	<tr>
		<td> cuda8-cudnn7-mkl </td>
		<td> <a href="https://paddle-wheel.bj.bcebos.com/1.5.0-gpu-cuda8-cudnn7-mkl/paddlepaddle_gpu-1.5.0.post87-cp27-cp27mu-linux_x86_64.whl"> paddlepaddle_gpu-1.5.0-cp27-cp27mu-linux_x86_64.whl</a></td>
		<td> <a href="https://paddle-wheel.bj.bcebos.com/1.5.0-gpu-cuda8-cudnn7-mkl/paddlepaddle_gpu-1.5.0.post87-cp27-cp27m-linux_x86_64.whl"> paddlepaddle_gpu-1.5.0-cp27-cp27m-linux_x86_64.whl</a></td>
		<td> <a href="https://paddle-wheel.bj.bcebos.com/1.5.0-gpu-cuda8-cudnn7-mkl/paddlepaddle_gpu-1.5.0.post87-cp35-cp35m-linux_x86_64.whl"> paddlepaddle_gpu-1.5.0-cp35-cp35m-linux_x86_64.whl</a></td>
		<td> <a href="https://paddle-wheel.bj.bcebos.com/1.5.0-gpu-cuda8-cudnn7-mkl/paddlepaddle_gpu-1.5.0.post87-cp36-cp36m-linux_x86_64.whl"> paddlepaddle_gpu-1.5.0-cp36-cp36m-linux_x86_64.whl</a></td>
		<td> <a href="https://paddle-wheel.bj.bcebos.com/1.5.0-gpu-cuda8-cudnn7-mkl/paddlepaddle_gpu-1.5.0.post87-cp37-cp37m-linux_x86_64.whl"> paddlepaddle_gpu-1.5.0-cp37-cp37m-linux_x86_64.whl</a></td>
	</tr>
	<tr>
		<td> cuda9-cudnn7-mkl </td>
		<td> <a href="https://paddle-wheel.bj.bcebos.com/1.5.0-gpu-cuda9-cudnn7-mkl/paddlepaddle_gpu-1.5.0.post97-cp27-cp27mu-linux_x86_64.whl"> paddlepaddle_gpu-1.5.0-cp27-cp27mu-linux_x86_64.whl</a></td>
		<td> <a href="https://paddle-wheel.bj.bcebos.com/1.5.0-gpu-cuda9-cudnn7-mkl/paddlepaddle_gpu-1.5.0.post97-cp27-cp27m-linux_x86_64.whl"> paddlepaddle_gpu-1.5.0-cp27-cp27m-linux_x86_64.whl</a></td>
		<td> <a href="https://paddle-wheel.bj.bcebos.com/1.5.0-gpu-cuda9-cudnn7-mkl/paddlepaddle_gpu-1.5.0.post97-cp35-cp35m-linux_x86_64.whl"> paddlepaddle_gpu-1.5.0-cp35-cp35m-linux_x86_64.whl</a></td>
		<td> <a href="https://paddle-wheel.bj.bcebos.com/1.5.0-gpu-cuda9-cudnn7-mkl/paddlepaddle_gpu-1.5.0.post97-cp36-cp36m-linux_x86_64.whl"> paddlepaddle_gpu-1.5.0-cp36-cp36m-linux_x86_64.whl</a></td>
		<td> <a href="https://paddle-wheel.bj.bcebos.com/1.5.0-gpu-cuda9-cudnn7-mkl/paddlepaddle_gpu-1.5.0.post97-cp37-cp37m-linux_x86_64.whl"> paddlepaddle_gpu-1.5.0-cp37-cp37m-linux_x86_64.whl</a></td>
	</tr>
	<tr>
		<td> cuda10_cudnn7-mkl </td>
		<td> <a href="https://paddle-wheel.bj.bcebos.com/1.5.0-gpu-cuda10-cudnn7-mkl/paddlepaddle_gpu-1.5.0.post107-cp27-cp27mu-linux_x86_64.whl"> paddlepaddle_gpu-1.5.0-cp27-cp27mu-linux_x86_64.whl</a></td>
		<td> <a href="https://paddle-wheel.bj.bcebos.com/1.5.0-gpu-cuda10-cudnn7-mkl/paddlepaddle_gpu-1.5.0.post107-cp27-cp27m-linux_x86_64.whl"> paddlepaddle_gpu-1.5.0-cp27-cp27m-linux_x86_64.whl</a></td>
		<td> <a href="https://paddle-wheel.bj.bcebos.com/1.5.0-gpu-cuda10-cudnn7-mkl/paddlepaddle_gpu-1.5.0.post107-cp35-cp35m-linux_x86_64.whl"> paddlepaddle_gpu-1.5.0-cp35-cp35m-linux_x86_64.whl</a></td>
		<td> <a href="https://paddle-wheel.bj.bcebos.com/1.5.0-gpu-cuda10-cudnn7-mkl/paddlepaddle_gpu-1.5.0.post107-cp36-cp36m-linux_x86_64.whl">
		paddlepaddle_gpu-1.5.0-cp36-cp36m-linux_x86_64.whl</a></td>
		<td> <a href="https://paddle-wheel.bj.bcebos.com/1.5.0-gpu-cuda10-cudnn7-mkl/paddlepaddle_gpu-1.5.0.post107-cp37-cp37m-linux_x86_64.whl">
		paddlepaddle_gpu-1.5.0-cp37-cp37m-linux_x86_64.whl</a></td>
	</tr>   
	<tr>
		<td> mac_cpu </td>
		<td> - </td>
		<td> <a href="https://paddle-wheel.bj.bcebos.com/1.5.0-cpu-mac/paddlepaddle-1.5.0-cp27-cp27m-macosx_10_6_intel.whl">
		paddlepaddle-1.5.0-cp27-cp27m-macosx_10_6_intel.whl</a></td>
		<td> <a href="https://paddle-wheel.bj.bcebos.com/1.5.0-cpu-mac/paddlepaddle-1.5.0-cp35-cp35m-macosx_10_6_intel.whl">
		paddlepaddle-1.5.0-cp35-cp35m-macosx_10_6_intel.whl</a></td>
		<td> <a href="https://paddle-wheel.bj.bcebos.com/1.5.0-cpu-mac/paddlepaddle-1.5.0-cp36-cp36m-macosx_10_6_intel.whl">
		paddlepaddle-1.5.0-cp36-cp36m-macosx_10_6_intel.whl</a></td>
		<td> <a href="https://paddle-wheel.bj.bcebos.com/1.5.0-cpu-mac/paddlepaddle-1.5.0-cp37-cp37m-macosx_10_6_intel.whl">
		paddlepaddle-1.5.0-cp37-cp37m-macosx_10_6_intel.whl</a></td>
	</tr>
   </tbody>
</table>
</p>


<a name="ciwhls"></a>
</br></br>

## **Multi-version whl package list - dev**


<p align="center">
<table>
	<thead>
	<tr>
		<th> Release Instruction </th>
		<th> cp27-cp27mu </th>
		<th> cp27-cp27m </th>
		<th> cp35-cp35m	</th>
		<th> cp36-cp36m	</th>
		<th> cp37-cp37m	</th>
	</tr>
	</thead>
	<tbody>
	<tr>
		<td> cpu-mkl </td>
		<td> <a href="https://paddle-wheel.bj.bcebos.com/latest-cpu-mkl/paddlepaddle-latest-cp27-cp27mu-linux_x86_64.whl">
		paddlepaddle-latest-cp27-cp27mu-linux_x86_64.whl</a></td>
		<td> <a href="https://paddle-wheel.bj.bcebos.com/latest-cpu-mkl/paddlepaddle-latest-cp27-cp27m-linux_x86_64.whl">
		paddlepaddle-latest-cp27-cp27m-linux_x86_64.whl</a></td>
		<td> <a href="https://paddle-wheel.bj.bcebos.com/latest-cpu-mkl/paddlepaddle-latest-cp35-cp35m-linux_x86_64.whl">
		paddlepaddle-latest-cp35-cp35m-linux_x86_64.whl</a></td>
		<td> <a href="https://paddle-wheel.bj.bcebos.com/latest-cpu-mkl/paddlepaddle-latest-cp36-cp36m-linux_x86_64.whl">
		paddlepaddle-latest-cp36-cp36m-linux_x86_64.whl</a></td>
		<td> <a href="https://paddle-wheel.bj.bcebos.com/latest-cpu-mkl/paddlepaddle-latest-cp37-cp37m-linux_x86_64.whl">
		paddlepaddle-latest-cp37-cp37m-linux_x86_64.whl</a></td>
	</tr>	
	<tr>
		<td> cpu-openblas </td>
		<td> <a href="https://paddle-wheel.bj.bcebos.com/latest-cpu-openblas/paddlepaddle-latest-cp27-cp27mu-linux_x86_64.whl">
		paddlepaddle-latest-cp27-cp27mu-linux_x86_64.whl</a></td>
		<td> <a href="https://paddle-wheel.bj.bcebos.com/latest-cpu-openblas/paddlepaddle-latest-cp27-cp27m-linux_x86_64.whl"> paddlepaddle-latest-cp27-cp27m-linux_x86_64.whl</a></td>
		<td> <a href="https://paddle-wheel.bj.bcebos.com/latest-cpu-openblas/paddlepaddle-latest-cp35-cp35m-linux_x86_64.whl">
		paddlepaddle-latest-cp35-cp35m-linux_x86_64.whl</a></td>
		<td> <a href="https://paddle-wheel.bj.bcebos.com/latest-cpu-openblas/paddlepaddle-latest-cp36-cp36m-linux_x86_64.whl">
		paddlepaddle-latest-cp36-cp36m-linux_x86_64.whl</a></td>
		<td> <a href="https://paddle-wheel.bj.bcebos.com/latest-cpu-openblas/paddlepaddle-latest-cp37-cp37m-linux_x86_64.whl">
		paddlepaddle-latest-cp37-cp37m-linux_x86_64.whl</a></td>
	</tr>
	<tr>
		<td> cuda8-cudnn7-openblas </td>
		<td> <a href="https://paddle-wheel.bj.bcebos.com/latest-gpu-cuda8-cudnn7-openblas/paddlepaddle_gpu-latest-cp27-cp27mu-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp27-cp27mu-linux_x86_64.whl</a></td>
		<td> <a href="https://paddle-wheel.bj.bcebos.com/latest-gpu-cuda8-cudnn7-openblas/paddlepaddle_gpu-latest-cp27-cp27m-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp27-cp27m-linux_x86_64.whl</a></td>
		<td> <a href="https://paddle-wheel.bj.bcebos.com/latest-gpu-cuda8-cudnn7-openblas/paddlepaddle_gpu-latest-cp35-cp35m-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp35-cp35m-linux_x86_64.whl</a></td>
		<td> <a href="https://paddle-wheel.bj.bcebos.com/latest-gpu-cuda8-cudnn7-openblas/paddlepaddle_gpu-latest-cp36-cp36m-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp36-cp36m-linux_x86_64.whl</a></td>
		<td> <a href="https://paddle-wheel.bj.bcebos.com/latest-gpu-cuda8-cudnn7-openblas/paddlepaddle_gpu-latest-cp37-cp37m-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp37-cp37m-linux_x86_64.whl</a></td>
	</tr>
	<tr>
		<td> cuda8-cudnn7-mkl </td>
		<td> <a href="https://paddle-wheel.bj.bcebos.com/latest-gpu-cuda8-cudnn7-mkl/paddlepaddle_gpu-latest-cp27-cp27mu-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp27-cp27mu-linux_x86_64.whl</a></td>
		<td> <a href="https://paddle-wheel.bj.bcebos.com/latest-gpu-cuda8-cudnn7-mkl/paddlepaddle_gpu-latest-cp27-cp27m-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp27-cp27m-linux_x86_64.whl</a></td>
		<td> <a href="https://paddle-wheel.bj.bcebos.com/latest-gpu-cuda8-cudnn7-mkl/paddlepaddle_gpu-latest-cp35-cp35m-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp35-cp35m-linux_x86_64.whl</a></td>
		<td> <a href="https://paddle-wheel.bj.bcebos.com/latest-gpu-cuda8-cudnn7-mkl/paddlepaddle_gpu-latest-cp36-cp36m-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp36-cp36m-linux_x86_64.whl</a></td>
		<td> <a href="https://paddle-wheel.bj.bcebos.com/latest-gpu-cuda8-cudnn7-mkl/paddlepaddle_gpu-latest-cp37-cp37m-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp37-cp37m-linux_x86_64.whl</a></td>
	</tr>
	<tr>
		<td> cuda9-cudnn7-mkl </td>
		<td> <a href="https://paddle-wheel.bj.bcebos.com/latest-gpu-cuda9-cudnn7-mkl/paddlepaddle_gpu-latest-cp27-cp27mu-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp27-cp27mu-linux_x86_64.whl</a></td>
		<td> <a href="https://paddle-wheel.bj.bcebos.com/latest-gpu-cuda9-cudnn7-mkl/paddlepaddle_gpu-latest-cp27-cp27m-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp27-cp27m-linux_x86_64.whl</a></td>
		<td> <a href="https://paddle-wheel.bj.bcebos.com/latest-gpu-cuda9-cudnn7-mkl/paddlepaddle_gpu-latest-cp35-cp35m-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp35-cp35m-linux_x86_64.whl</a></td>
		<td> <a href="https://paddle-wheel.bj.bcebos.com/latest-gpu-cuda9-cudnn7-mkl/paddlepaddle_gpu-latest-cp36-cp36m-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp36-cp36m-linux_x86_64.whl</a></td>
		<td> <a href="https://paddle-wheel.bj.bcebos.com/latest-gpu-cuda9-cudnn7-mkl/paddlepaddle_gpu-latest-cp37-cp37m-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp37-cp37m-linux_x86_64.whl</a></td>
	</tr>
	<tr>
		<td> cuda10_cudnn7-mkl </td>
		<td> <a href="https://paddle-wheel.bj.bcebos.com/latest-gpu-cuda10-cudnn7-mkl/paddlepaddle_gpu-latest-cp27-cp27mu-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp27-cp27mu-linux_x86_64.whl</a></td>
		<td> <a href="https://paddle-wheel.bj.bcebos.com/latest-gpu-cuda10-cudnn7-mkl/paddlepaddle_gpu-latest-cp27-cp27m-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp27-cp27m-linux_x86_64.whl</a></td>
		<td> <a href="https://paddle-wheel.bj.bcebos.com/latest-gpu-cuda10-cudnn7-mkl/paddlepaddle_gpu-latest-cp35-cp35m-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp35-cp35m-linux_x86_64.whl</a></td>
		<td> <a href="https://paddle-wheel.bj.bcebos.com/latest-gpu-cuda10-cudnn7-mkl/paddlepaddle_gpu-latest-cp36-cp36m-linux_x86_64.whl">
		paddlepaddle_gpu-latest-cp36-cp36m-linux_x86_64.whl</a></td>
		<td> <a href="https://paddle-wheel.bj.bcebos.com/latest-gpu-cuda10-cudnn7-mkl/paddlepaddle_gpu-latest-cp37-cp37m-linux_x86_64.whl">
		paddlepaddle_gpu-latest-cp37-cp37m-linux_x86_64.whl</a></td>
	</tr>
   </tbody>
</table>
</p>


</br></br>

## Execute the PaddlePaddle training program in Docker

***

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

***

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

***

In order to ensure that the GPU driver works properly in the image, we recommend using [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) to run the image. Don't forget to install the latest GPU drivers on your physical machine in advance.

`Nvidia-docker run -it -v $PWD:/work hub.baidubce.com/paddlepaddle/paddle:latest-gpu /bin/bash`

**Note: If you don't have nvidia-docker installed, you can try the following to mount the CUDA library and Linux devices into the Docker container:**


	export CUDA_SO="$(\ls /usr/lib64/libcuda* | xargs -I{} echo '-v {}:{}') \
	$(\ls /usr/lib64/libnvidia* | xargs -I{} echo '-v {}:{}')"
	export DEVICES=$(\ls /dev/nvidia* | xargs -I{} echo '--device {}:{}')
	docker run ${CUDA_SO} \
 	 ${DEVICES} -it hub.baidubce.com/paddlepaddle/paddle:latest-gpu



**About AVX:**

AVX is a set of CPU instructions that speeds up the calculation of PaddlePaddle. The latest PaddlePaddle Docker image is enabled by default for AVX compilation, so if your computer does not support AVX, you need to compile PaddlePaddle to no-avx version separately.

The following instructions can check if the Linux computer supports AVX:

`if cat /proc/cpuinfo | grep -i avx; then echo Yes; else echo No; fi`

If the output is No, you need to choose a mirror that uses no-AVX.




























