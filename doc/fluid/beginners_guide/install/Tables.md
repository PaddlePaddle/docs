***
<a name="third_party"></a>
# 附录

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
		<td> 3.4 </td>
		<td>  </td>
		<td>  </td>
	</tr>
	<tr>
		<td> GCC </td>
		<td> 4.8 / 5.4 </td>
		<td>  推荐使用CentOS的devtools2 </td>
		<td>  </td>
	</tr>
		<tr>
		<td> Python </td>
		<td> 2.7.x. </td>
		<td> 依赖libpython2.7.so </td>
		<td> <code> apt install python-dev </code> 或 <code> yum install python-devel </code></td>
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
		<td>  </td>
		<td>  </td>
	</tr>
	<tr>
		<td> pip </td>
		<td> 最低9.0.1 </td>
		<td>  </td>
		<td> <code> apt install python-pip </code> 或 <code> yum install Python-pip </code> </td>
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
		<td> <code> apt install patchelf </code> 或参见github <a href="https://gist.github.com/ruario/80fefd174b3395d34c14">patchELF 官方文档</a></td>
	</tr>
	<tr>
		<td> go </td>
		<td> >=1.8 </td>
		<td> 可选 </td>
		<td>  </td>
	</tr>
	</tbody>
</table>
</p>


***
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
		<td> 是否支持GPU </td>
		<td> ON </td>
	</tr>
	<tr>
		<td> WITH_C_API </td>
		<td> 是否仅编译CAPI </td>
		<td>  OFF </td>
	</tr>
		<tr>
		<td> WITH_DOUBLE </td>
		<td> 是否使用双精度浮点数 </td>
		<td> OFF </td>
	</tr>
	<tr>
		<td> WITH_DSO </td>
		<td> 是否运行时动态加载CUDA动态库，而非静态加载CUDA动态库 </td>
		<td> ON </td>
	</tr>
	<tr>
		<td> WITH_AVX </td>
		<td> 是否编译含有AVX指令集的PaddlePaddle二进制文件 </td>
		<td> ON </td>
	</tr>
	<tr>
		<td> WITH_PYTHON </td>
		<td> 是否内嵌PYTHON解释器 </td>
		<td> ON </td>
	</tr>
	<tr>
		<td> WITH_STYLE_CHECK </td>
		<td> 是否编译时进行代码风格检查 </td>
		<td> ON </td>
	</tr>
	<tr>
		<td> WITH_TESTING </td>
		<td> 是否开启单元测试 </td>
		<td> OFF </td>
	</tr>
	<tr>
		<td> WITH_DOC </td>
		<td> 是否编译中英文文档 </td>
		<td> OFF </td>
	</tr>
	<tr>
		<td> WITH_SWIG_PY </td>
		<td> 是否编译PYTHON的SWIG接口，该接口可用于预测和定制化训练 </td>
		<td> Auto </td>
	<tr>
		<td> WITH_GOLANG </td>
		<td> 是否编译go语言的可容错parameter server </td>
		<td> OFF </td>
	</tr>
	<tr>
		<td> WITH_MKL </td>
		<td> 是否使用MKL数学库，如果为否则是用OpenBLAS </td>
		<td> ON </td>
	</tr>
	<tr>
		<td> WITH_SYSTEM_BLAS </td>
		<td> 是否使用系统自带的BLAS </td>
		<td> OFF </td>
	</tr>
	<tr>
		<td> WITH_DISTRIBUTE </td>
		<td> 是否编译带有分布式的版本 </td>
		<td> OFF </td>
	</tr>
	<tr>
		<td> WITH_MKL </td>
		<td> 是否使用MKL数学库，如果为否则是用OpenBLAS </td>
		<td> ON </td>
	</tr>
	<tr>
		<td> WITH_RDMA </td>
		<td> 是否编译支持RDMA的相关部分 </td>
		<td> OFF </td>
	</tr>
	<tr>
		<td> WITH_BRPC_RDMA </td>
		<td> 是否使用BRPC RDMA作为RPC协议 </td>
		<td> OFF </td>
	</tr>
		<tr>
		<td> ON_INFER </td>
		<td> 是否打开预测优化 </td>
		<td> OFF </td>
	</tr>
	<tr>
		<td> DWITH_ANAKIN </td>
		<td> 是否编译ANAKIN </td>
		<td> OFF </td>
	</tr>
   </tbody>
</table>
</p>





**BLAS**

PaddlePaddle支持 [MKL](https://software.intel.com/en-us/mkl) 和 [OpenBlAS](http://www.openblas.net) 两种BLAS库。默认使用MKL。如果使用MKL并且机器含有AVX2指令集，还会下载MKL-DNN数学库，详细参考[这里](https://github.com/PaddlePaddle/Paddle/tree/develop/doc/design/mkldnn#cmake) 。

如果关闭MKL，则会使用OpenBLAS作为BLAS库。

**CUDA/cuDNN**

PaddlePaddle在编译时/运行时会自动找到系统中安装的CUDA和cuDNN库进行编译和执行。 使用参数 `-DCUDA_ARCH_NAME=Auto` 可以指定开启自动检测SM架构，加速编译。

PaddlePaddle可以使用cuDNN v5.1之后的任何一个版本来编译运行，但尽量请保持编译和运行使用的cuDNN是同一个版本。 我们推荐使用最新版本的cuDNN。

**编译选项的设置**

PaddePaddle通过编译时指定路径来实现引用各种BLAS/CUDA/cuDNN库。cmake编译时，首先在系统路径（ `/usr/liby` 和 `/usr/local/lib` ）中搜索这几个库，同时也会读取相关路径变量来进行搜索。 通过使用`-D`命令可以设置，例如：

> `cmake .. -DWITH_GPU=ON -DWITH_TESTING=OFF -DCUDNN_ROOT=/opt/cudnnv5`

**注意**：这几个编译选项的设置，只在第一次cmake的时候有效。如果之后想要重新设置，推荐清理整个编译目录（ rm -rf ）后，再指定。


***
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
		<td> paddlepaddle==[版本号] 如 paddlepaddle==1.0.1(下载1.0.1版本只支持CPU的PaddlePaddle)</td>
		<td> 只支持CPU对应版本的PaddlePaddle，具体版本请参见<a href=https://pypi.org/project/paddlepaddle/#history>Pypi</a> </td>
	</tr>
	<tr>
		<td> paddlepaddle-gpu==1.0.1 </td>
		<td> 使用CUDA 9.0和cuDNN 7编译的1.0.1版本 </td>
	</tr>
	<tr>
		<td> paddlepaddle-gpu==1.0.1.post87 </td>
		<td> 使用CUDA 8.0和cuDNN 7编译的1.0.1版本 </td>
	</tr>
	<tr>
		<td> paddlepaddle-gpu==1.0.1.post85 </td>
		<td> 使用CUDA 8.0和cuDNN 5编译的1.0.1版本 </td>
	</tr>
	<tr>
		<td> paddlepaddle-gpu==1.0.0 </td>
		<td> 使用CUDA 9.0和cuDNN 7编译的1.0.0版本 </td>
	</tr>
	<tr>
		<td> paddlepaddle-gpu==1.0.0.post87 </td>
		<td> 使用CUDA 8.0和cuDNN 7编译的1.0.0版本 </td>
	</tr>
	<tr>
		<td> paddlepaddle-gpu==1.0.0.post85 </td>
		<td> 使用CUDA 8.0和cuDNN 5编译的1.0.0版本 </td>
	</tr>
	<tr>
		<td> paddlepaddle-gpu==0.15.0 </td>
		<td> 使用CUDA 9.0和cuDNN 7编译的0.15.0版本 </td>
	</tr>
	<tr>
		<td> paddlepaddle-gpu==0.15.0.post87 </td>
		<td> 使用CUDA 8.0和cuDNN 7编译的0.15.0版本 </td>
	</tr>
	<tr>
		<td> paddlepaddle-gpu==0.15.0.post85 </td>
		<td> 使用CUDA 8.0和cuDNN 5编译的0.15.0版本 </td>
	</tr>
	<tr>
		<td> paddlepaddle-gpu==0.14.0 </td>
		<td> 使用CUDA 9.0和cuDNN 7编译的0.15.0版本 </td>
	</tr>
	<tr>
		<td> paddlepaddle-gpu==0.14.0.post87 </td>
		<td> 使用CUDA 8.0和cuDNN 7编译的0.15.0版本 </td>
	</tr>
	<tr>
		<td> paddlepaddle-gpu==0.14.0.post85 </td>
		<td> 使用CUDA 8.0和cuDNN 5编译的0.15.0版本 </td>
	</tr>
	<tr>
		<td> paddlepaddle-gpu==0.13.0 </td>
		<td> 使用CUDA 9.0和cuDNN 7编译的0.13.0版本 </td>
	</tr>
	<tr>
		<td> paddlepaddle-gpu==0.12.0 </td>
		<td> 使用CUDA 8.0和cuDNN 5编译的0.12.0版本 </td>
	</tr>
	<tr>
		<td> paddlepaddle-gpu==0.11.0.post87 </td>
		<td> 使用CUDA 8.0和cuDNN 7编译的0.11.0版本 </td>
	</tr>
	<tr>
		<td> paddlepaddle-gpu==0.11.0.post85 </td>
		<td> 使用CUDA 8.0和cuDNN 5编译的0.11.0版本 </td>
	</tr>
	<tr>
		<td> paddlepaddle-gpu==0.11.0 </td>
		<td> 使用CUDA 7.5和cuDNN 5编译的0.11.0版本 </td>
	</tr>
   </tbody>
</table>
</p>


您可以在 [Release History](https://pypi.org/project/paddlepaddle-gpu/#history) 中找到PaddlePaddle-gpu的各个发行版本。

***
<a name="dockers"></a>
</br></br>
## **安装镜像表及简介**
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
		<td> hub.baidubce.com/paddlepaddle/paddle:latest </td>
		<td> 最新的预先安装好PaddlePaddle CPU版本的镜像 </td>
	</tr>
	<tr>
		<td> hub.baidubce.com/paddlepaddle/paddle:latest-dev </td>
		<td> 最新的PaddlePaddle的开发环境 </td>
	</tr>
		<tr>
		<td> hub.baidubce.com/paddlepaddle/paddle:[Version] </td>
		<td> 将version换成具体的版本，历史版本的预安装好PaddlePaddle的镜像 </td>
	</tr>
	<tr>
		<td> hub.baidubce.com/paddlepaddle/paddle:latest-gpu </td>
		<td> 最新的预先安装好PaddlePaddle GPU版本的镜像 </td>
	</tr>
   </tbody>
</table>
</p>


您可以在 [DockerHub](https://hub.docker.com/r/paddlepaddle/paddle/tags/) 中找到PaddlePaddle的各个发行的版本的docker镜像。

***

<a name="ciwhls-release"></a>
</br></br>

## **多版本whl包列表-Release**

<p align="center">
<table>
	<thead>
	<tr>
		<th> 版本说明 </th>
		<th> cp27-cp27mu </th>
		<th> cp27-cp27m </th>
		<th> cp35-cp35m	</th>
		<th> cp36-cp36m	</th>
		<th> cp37-cp37m	</th>
	</tr>
	</thead>
	<tbody>
	<tr>
		<td> cpu-noavx-mkl </td>
		<td> <a href="http://paddlepaddle.org/download?url=http://paddle-wheel.bj.bcebos.com/1.2.0-cpu-noavx-mkl/paddlepaddle-1.2.0-cp27-cp27mu-linux_x86_64.whl">
		paddlepaddle-1.2.0-cp27-cp27mu-linux_x86_64.whl</a></td>
		<td> <a href="http://paddlepaddle.org/download?url=http://paddle-wheel.bj.bcebos.com/1.2.0-cpu-noavx-mkl/paddlepaddle-1.2.0-cp27-cp27m-linux_x86_64.whl">
		paddlepaddle-1.2.0-cp27-cp27m-linux_x86_64.whl</a></td>
		<td> <a href="http://paddlepaddle.org/download?url=http://paddle-wheel.bj.bcebos.com/1.2.0-cpu-noavx-mkl/paddlepaddle-1.2.0-cp35-cp35m-linux_x86_64.whl">
		paddlepaddle-1.2.0-cp35-cp35m-linux_x86_64.whl</a></td>
		<td> <a href="http://paddlepaddle.org/download?url=http://paddle-wheel.bj.bcebos.com/1.2.0-cpu-noavx-mkl/paddlepaddle-1.2.0-cp36-cp36m-linux_x86_64.whl">
		paddlepaddle-1.2.0-cp36-cp36m-linux_x86_64.whl</a></td>
		<td> <a href="http://paddlepaddle.org/download?url=http://paddle-wheel.bj.bcebos.com/1.2.0-cpu-noavx-mkl/paddlepaddle-1.2.0-cp37-cp37m-linux_x86_64.whl">
		paddlepaddle-1.2.0-cp37-cp37m-linux_x86_64.whl</a></td>
	</tr>
	<tr>
		<td> cpu_avx_mkl </td>
		<td> <a href="http://paddlepaddle.org/download?url=http://paddle-wheel.bj.bcebos.com/1.2.0-cpu-avx-mkl/paddlepaddle-1.2.0-cp27-cp27mu-linux_x86_64.whl">
		paddlepaddle-1.2.0-cp27-cp27mu-linux_x86_64.whl</a></td>
		<td> <a href="http://paddlepaddle.org/download?url=http://paddle-wheel.bj.bcebos.com/1.2.0-cpu-avx-mkl/paddlepaddle-1.2.0-cp27-cp27m-linux_x86_64.whl">
		paddlepaddle-1.2.0-cp27-cp27m-linux_x86_64.whl</a></td>
		<td> <a href="http://paddlepaddle.org/download?url=http://paddle-wheel.bj.bcebos.com/1.2.0-cpu-avx-mkl/paddlepaddle-1.2.0-cp35-cp35m-linux_x86_64.whl">
		paddlepaddle-1.2.0-cp35-cp35m-linux_x86_64.whl</a></td>
		<td> <a href="http://paddlepaddle.org/download?url=http://paddle-wheel.bj.bcebos.com/1.2.0-cpu-avx-mkl/paddlepaddle-1.2.0-cp36-cp36m-linux_x86_64.whl">
		paddlepaddle-1.2.0-cp36-cp36m-linux_x86_64.whl</a></td>
		<td> <a href="http://paddlepaddle.org/download?url=http://paddle-wheel.bj.bcebos.com/1.2.0-cpu-avx-mkl/paddlepaddle-1.2.0-cp37-cp37m-linux_x86_64.whl">
		paddlepaddle-1.2.0-cp37-cp37m-linux_x86_64.whl</a></td>
	</tr>
	<tr>
		<td> cpu_avx_openblas </td>
		<td> <a href="http://paddlepaddle.org/download?url=http://paddle-wheel.bj.bcebos.com/1.2.0-cpu-avx-openblas/paddlepaddle-1.2.0-cp27-cp27mu-linux_x86_64.whl">
		paddlepaddle-1.2.0-cp27-cp27mu-linux_x86_64.whl</a></td>
		<td> <a href="http://paddlepaddle.org/download?url=http://paddle-wheel.bj.bcebos.com/1.2.0-cpu-avx-openblas/paddlepaddle-1.2.0-cp27-cp27m-linux_x86_64.whl"> paddlepaddle-1.2.0-cp27-cp27m-linux_x86_64.whl</a></td>
		<td> <a href="http://paddlepaddle.org/download?url=http://paddle-wheel.bj.bcebos.com/1.2.0-cpu-avx-openblas/paddlepaddle-1.2.0-cp35-cp35m-linux_x86_64.whl">
		paddlepaddle-1.2.0-cp35-cp35m-linux_x86_64.whl</a></td>
		<td> <a href="http://paddlepaddle.org/download?url=http://paddle-wheel.bj.bcebos.com/1.2.0-cpu-avx-openblas/paddlepaddle-1.2.0-cp36-cp36m-linux_x86_64.whl">
		paddlepaddle-1.2.0-cp36-cp36m-linux_x86_64.whl</a></td>
		<td> <a href="http://paddlepaddle.org/download?url=http://paddle-wheel.bj.bcebos.com/1.2.0-cpu-avx-openblas/paddlepaddle-1.2.0-cp37-cp37m-linux_x86_64.whl">
		paddlepaddle-1.2.0-cp37-cp37m-linux_x86_64.whl</a></td>
	</tr>
	<tr>
		<td> cuda8.0_cudnn5_avx_mkl </td>
		<td> <a href="http://paddlepaddle.org/download?url=http://paddle-wheel.bj.bcebos.com/1.2.0-gpu-cuda8-cudnn5-avx-mkl/paddlepaddle_gpu-1.2.0.post85-cp27-cp27mu-linux_x86_64.whl"> paddlepaddle_gpu-1.2.0-cp27-cp27mu-linux_x86_64.whl</a></td>
		<td><a href="http://paddlepaddle.org/download?url=http://paddle-wheel.bj.bcebos.com/1.2.0-gpu-cuda8-cudnn5-avx-mkl/paddlepaddle_gpu-1.2.0.post85-cp27-cp27m-linux_x86_64.whl"> paddlepaddle_gpu-1.2.0-cp27-cp27m-linux_x86_64.whl</a></td>
		<td><a href="http://paddlepaddle.org/download?url=http://paddle-wheel.bj.bcebos.com/1.2.0-gpu-cuda8-cudnn5-avx-mkl/paddlepaddle_gpu-1.2.0.post85-cp35-cp35m-linux_x86_64.whl"> paddlepaddle_gpu-1.2.0-cp35-cp35m-linux_x86_64.whl</a></td>
		<td> <a href="http://paddlepaddle.org/download?url=http://paddle-wheel.bj.bcebos.com/1.2.0-gpu-cuda8-cudnn5-avx-mkl/paddlepaddle_gpu-1.2.0.post85-cp36-cp36m-linux_x86_64.whl">
		paddlepaddle_gpu-1.2.0-cp36-cp36m-linux_x86_64.whl</a></td>
		<td> <a href="http://paddlepaddle.org/download?url=http://paddle-wheel.bj.bcebos.com/1.2.0-gpu-cuda8-cudnn5-avx-mkl/paddlepaddle_gpu-1.2.0.post85-cp37-cp37m-linux_x86_64.whl">
		paddlepaddle_gpu-1.2.0-cp37-cp37m-linux_x86_64.whl</a></td>
	</tr>
	<tr>
		<td> cuda8.0_cudnn7_noavx_mkl </td>
		<td> <a href="http://paddlepaddle.org/download?url=http://paddle-wheel.bj.bcebos.com/1.2.0-gpu-cuda8-cudnn7-noavx-mkl/paddlepaddle_gpu-1.2.0-cp27-cp27mu-linux_x86_64.whl"> paddlepaddle_gpu-1.2.0-cp27-cp27mu-linux_x86_64.whl</a></td>
		<td><a href="http://paddlepaddle.org/download?url=http://paddle-wheel.bj.bcebos.com/1.2.0-gpu-cuda8-cudnn7-noavx-mkl/paddlepaddle_gpu-1.2.0-cp27-cp27m-linux_x86_64.whl"> paddlepaddle_gpu-1.2.0-cp27-cp27m-linux_x86_64.whl</a></td>
		<td><a href="http://paddlepaddle.org/download?url=http://paddle-wheel.bj.bcebos.com/1.2.0-gpu-cuda8-cudnn7-noavx-mkl/paddlepaddle_gpu-1.2.0-cp35-cp35m-linux_x86_64.whl"> paddlepaddle_gpu-1.2.0-cp35-cp35m-linux_x86_64.whl</a></td>
		<td> <a href="http://paddlepaddle.org/download?url=http://paddle-wheel.bj.bcebos.com/1.2.0-gpu-cuda8-cudnn7-noavx-mkl/paddlepaddle_gpu-1.2.0-cp36-cp36m-linux_x86_64.whl">
		paddlepaddle_gpu-1.2.0-cp36-cp36m-linux_x86_64.whl</a></td>
		<td> <a href="http://paddlepaddle.org/download?url=http://paddle-wheel.bj.bcebos.com/1.2.0-gpu-cuda8-cudnn7-noavx-mkl/paddlepaddle_gpu-1.2.0-cp37-cp37m-linux_x86_64.whl">
		paddlepaddle_gpu-1.2.0-cp37-cp37m-linux_x86_64.whl</a></td>
	</tr>
	<tr>
		<td> cuda8.0_cudnn7_avx_mkl </td>
		<td> <a href="http://paddlepaddle.org/download?url=http://paddle-wheel.bj.bcebos.com/1.2.0-gpu-cuda8-cudnn7-avx-mkl/paddlepaddle_gpu-1.2.0.post87-cp27-cp27mu-linux_x86_64.whl"> paddlepaddle_gpu-1.2.0.post87-cp27-cp27mu-linux_x86_64.whl</a></td>
		<td><a href="http://paddlepaddle.org/download?url=http://paddle-wheel.bj.bcebos.com/1.2.0-gpu-cuda8-cudnn7-avx-mkl/paddlepaddle_gpu-1.2.0.post87-cp27-cp27m-linux_x86_64.whl"> paddlepaddle_gpu-1.2.0.post87-cp27-cp27m-linux_x86_64.whl</a></td>
		<td><a href="http://paddlepaddle.org/download?url=http://paddle-wheel.bj.bcebos.com/1.2.0-gpu-cuda8-cudnn7-avx-mkl/paddlepaddle_gpu-1.2.0.post87-cp35-cp35m-linux_x86_64.whl"> paddlepaddle_gpu-1.2.0.post87-cp35-cp35m-linux_x86_64.whl</a></td>
		<td> <a href="http://paddlepaddle.org/download?url=http://paddle-wheel.bj.bcebos.com/1.2.0-gpu-cuda8-cudnn7-avx-mkl/paddlepaddle_gpu-1.2.0.post87-cp36-cp36m-linux_x86_64.whl">
		paddlepaddle_gpu-1.2.0.post87-cp36-cp36m-linux_x86_64.whl</a></td>
		<td> <a href="http://paddlepaddle.org/download?url=http://paddle-wheel.bj.bcebos.com/1.2.0-gpu-cuda8-cudnn7-avx-mkl/paddlepaddle_gpu-1.2.0.post87-cp37-cp37m-linux_x86_64.whl">
		paddlepaddle_gpu-1.2.0.post87-cp37-cp37m-linux_x86_64.whl</a></td>
	</tr>
	<tr>
		<td> cuda9.0_cudnn7_avx_mkl </td>
		<td> <a href="http://paddlepaddle.org/download?url=http://paddle-wheel.bj.bcebos.com/1.2.0-gpu-cuda9-cudnn7-avx-mkl/paddlepaddle_gpu-1.2.0.post97-cp27-cp27mu-linux_x86_64.whl"> paddlepaddle_gpu-1.2.0-cp27-cp27mu-linux_x86_64.whl</a></td>
		<td><a href="http://paddlepaddle.org/download?url=http://paddle-wheel.bj.bcebos.com/1.2.0-gpu-cuda9-cudnn7-avx-mkl/paddlepaddle_gpu-1.2.0.post97-cp27-cp27m-linux_x86_64.whl"> paddlepaddle_gpu-1.2.0-cp27-cp27m-linux_x86_64.whl</a></td>
		<td><a href="http://paddlepaddle.org/download?url=http://paddle-wheel.bj.bcebos.com/1.2.0-gpu-cuda9-cudnn7-avx-mkl/paddlepaddle_gpu-1.2.0.post97-cp35-cp35m-linux_x86_64.whl"> paddlepaddle_gpu-1.2.0-cp35-cp35m-linux_x86_64.whl</a></td>
		<td> <a href="http://paddlepaddle.org/download?url=http://paddle-wheel.bj.bcebos.com/1.2.0-gpu-cuda9-cudnn7-avx-mkl/paddlepaddle_gpu-1.2.0.post97-cp36-cp36m-linux_x86_64.whl">
		paddlepaddle_gpu-1.2.0-cp36-cp36m-linux_x86_64.whl</a></td>
		<td> <a href="http://paddlepaddle.org/download?url=http://paddle-wheel.bj.bcebos.com/1.2.0-gpu-cuda9-cudnn7-avx-mkl/paddlepaddle_gpu-1.2.0.post97-cp37-cp37m-linux_x86_64.whl">
		paddlepaddle_gpu-1.2.0-cp37-cp37m-linux_x86_64.whl</a></td>
	</tr>
   </tbody>
</table>
</p>

<a name="ciwhls"></a>
</br></br>
## **多版本whl包列表-dev**
<p align="center">
<table>
	<thead>
	<tr>
		<th> 版本说明 </th>
		<th> cp27-cp27mu </th>
		<th> cp27-cp27m </th>
		<th> cp35-cp35m	</th>
		<th> cp36-cp36m	</th>
		<th> cp37-cp37m	</th>
	</tr>
	</thead>
	<tbody>
	<tr>
		<td> cpu-noavx-mkl </td>
		<td> <a href="http://paddlepaddle.org/download?url=http://paddle-wheel.bj.bcebos.com/latest-cpu-noavx-mkl/paddlepaddle-latest-cp27-cp27mu-linux_x86_64.whl">
		paddlepaddle-latest-cp27-cp27mu-linux_x86_64.whl</a></td>
		<td> <a href="http://paddlepaddle.org/download?url=http://paddle-wheel.bj.bcebos.com/latest-cpu-noavx-mkl/paddlepaddle-latest-cp27-cp27m-linux_x86_64.whl">
		paddlepaddle-latest-cp27-cp27m-linux_x86_64.whl</a></td>
		<td> <a href="http://paddlepaddle.org/download?url=http://paddle-wheel.bj.bcebos.com/latest-cpu-noavx-mkl/paddlepaddle-latest-cp35-cp35m-linux_x86_64.whl">
		paddlepaddle-latest-cp35-cp35m-linux_x86_64.whl</a></td>
		<td> <a href="http://paddlepaddle.org/download?url=http://paddle-wheel.bj.bcebos.com/latest-cpu-noavx-mkl/paddlepaddle-latest-cp36-cp36m-linux_x86_64.whl">
		paddlepaddle-latest-cp36-cp36m-linux_x86_64.whl</a></td>
		<td> <a href="http://paddlepaddle.org/download?url=http://paddle-wheel.bj.bcebos.com/latest-cpu-noavx-mkl/paddlepaddle-latest-cp37-cp37m-linux_x86_64.whl">
		paddlepaddle-latest-cp37-cp37m-linux_x86_64.whl</a></td>
	</tr>
	<tr>
		<td> cpu_avx_mkl </td>
		<td> <a href="http://paddlepaddle.org/download?url=http://paddle-wheel.bj.bcebos.com/latest-cpu-avx-mkl/paddlepaddle-latest-cp27-cp27mu-linux_x86_64.whl">
		paddlepaddle-latest-cp27-cp27mu-linux_x86_64.whl</a></td>
		<td> <a href="http://paddlepaddle.org/download?url=http://paddle-wheel.bj.bcebos.com/latest-cpu-avx-mkl/paddlepaddle-latest-cp27-cp27m-linux_x86_64.whl">
		paddlepaddle-latest-cp27-cp27m-linux_x86_64.whl</a></td>
		<td> <a href="http://paddlepaddle.org/download?url=http://paddle-wheel.bj.bcebos.com/latest-cpu-avx-mkl/paddlepaddle-latest-cp35-cp35m-linux_x86_64.whl">
		paddlepaddle-latest-cp35-cp35m-linux_x86_64.whl</a></td>
		<td> <a href="http://paddlepaddle.org/download?url=http://paddle-wheel.bj.bcebos.com/latest-cpu-avx-mkl/paddlepaddle-latest-cp36-cp36m-linux_x86_64.whl">
		paddlepaddle-latest-cp36-cp36m-linux_x86_64.whl</a></td>
		<td> <a href="http://paddlepaddle.org/download?url=http://paddle-wheel.bj.bcebos.com/latest-cpu-avx-mkl/paddlepaddle-latest-cp37-cp37m-linux_x86_64.whl">
		paddlepaddle-latest-cp37-cp37m-linux_x86_64.whl</a></td>
	</tr>
	<tr>
		<td> cpu_avx_openblas </td>
		<td> <a href="http://paddlepaddle.org/download?url=http://paddle-wheel.bj.bcebos.com/latest-cpu-avx-openblas/paddlepaddle-latest-cp27-cp27mu-linux_x86_64.whl">
		paddlepaddle-latest-cp27-cp27mu-linux_x86_64.whl</a></td>
		<td> <a href="http://paddlepaddle.org/download?url=http://paddle-wheel.bj.bcebos.com/latest-cpu-avx-openblas/paddlepaddle-latest-cp27-cp27m-linux_x86_64.whl"> paddlepaddle-latest-cp27-cp27m-linux_x86_64.whl</a></td>
		<td> <a href="http://paddlepaddle.org/download?url=http://paddle-wheel.bj.bcebos.com/latest-cpu-avx-openblas/paddlepaddle-latest-cp35-cp35m-linux_x86_64.whl">
		paddlepaddle-latest-cp35-cp35m-linux_x86_64.whl</a></td>
		<td> <a href="http://paddlepaddle.org/download?url=http://paddle-wheel.bj.bcebos.com/latest-cpu-avx-openblas/paddlepaddle-latest-cp36-cp36m-linux_x86_64.whl">
		paddlepaddle-latest-cp36-cp36m-linux_x86_64.whl</a></td>
		<td> <a href="http://paddlepaddle.org/download?url=http://paddle-wheel.bj.bcebos.com/latest-cpu-avx-openblas/paddlepaddle-latest-cp37-cp37m-linux_x86_64.whl">
		paddlepaddle-latest-cp37-cp37m-linux_x86_64.whl</a></td>
	</tr>
	<tr>
		<td> cuda8.0_cudnn5_avx_mkl </td>
		<td> <a href="http://paddlepaddle.org/download?url=http://paddle-wheel.bj.bcebos.com/latest-gpu-cuda8-cudnn5-avx-mkl/paddlepaddle_gpu-latest-cp27-cp27mu-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp27-cp27mu-linux_x86_64.whl</a></td>
		<td><a href="http://paddlepaddle.org/download?url=http://paddle-wheel.bj.bcebos.com/latest-gpu-cuda8-cudnn5-avx-mkl/paddlepaddle_gpu-latest-cp27-cp27m-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp27-cp27m-linux_x86_64.whl</a></td>
		<td><a href="http://paddlepaddle.org/download?url=http://paddle-wheel.bj.bcebos.com/latest-gpu-cuda8-cudnn5-avx-mkl/paddlepaddle_gpu-latest-cp35-cp35m-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp35-cp35m-linux_x86_64.whl</a></td>
		<td> <a href="http://paddlepaddle.org/download?url=http://paddle-wheel.bj.bcebos.com/latest-gpu-cuda8-cudnn5-avx-mkl/paddlepaddle_gpu-latest-cp36-cp36m-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp36-cp36m-linux_x86_64.whl</a></td>
		<td> <a href="http://paddlepaddle.org/download?url=http://paddle-wheel.bj.bcebos.com/latest-gpu-cuda8-cudnn5-avx-mkl/paddlepaddle_gpu-latest-cp37-cp37m-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp37-cp37m-linux_x86_64.whl</a></td>
	</tr>
	<tr>
		<td> cuda8.0_cudnn7_noavx_mkl </td>
		<td> <a href="http://paddlepaddle.org/download?url=http://paddle-wheel.bj.bcebos.com/latest-gpu-cuda8-cudnn7-noavx-mkl/paddlepaddle_gpu-latest-cp27-cp27mu-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp27-cp27mu-linux_x86_64.whl</a></td>
		<td><a href="http://paddlepaddle.org/download?url=http://paddle-wheel.bj.bcebos.com/latest-gpu-cuda8-cudnn7-noavx-mkl/paddlepaddle_gpu-latest-cp27-cp27m-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp27-cp27m-linux_x86_64.whl</a></td>
		<td><a href="http://paddlepaddle.org/download?url=http://paddle-wheel.bj.bcebos.com/latest-gpu-cuda8-cudnn7-noavx-mkl/paddlepaddle_gpu-latest-cp35-cp35m-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp35-cp35m-linux_x86_64.whl</a></td>
		<td> <a href="http://paddlepaddle.org/download?url=http://paddle-wheel.bj.bcebos.com/latest-gpu-cuda8-cudnn7-noavx-mkl/paddlepaddle_gpu-latest-cp36-cp36m-linux_x86_64.whl">
		paddlepaddle_gpu-latest-cp36-cp36m-linux_x86_64.whl</a></td>
		<td> <a href="http://paddlepaddle.org/download?url=http://paddle-wheel.bj.bcebos.com/latest-gpu-cuda8-cudnn7-noavx-mkl/paddlepaddle_gpu-latest-cp37-cp37m-linux_x86_64.whl">
		paddlepaddle_gpu-latest-cp37-cp37m-linux_x86_64.whl</a></td>
	</tr>
	<tr>
		<td> cuda8.0_cudnn7_avx_mkl </td>
		<td> <a href="http://paddlepaddle.org/download?url=http://paddle-wheel.bj.bcebos.com/latest-gpu-cuda8-cudnn7-avx-mkl/paddlepaddle_gpu-latest-cp27-cp27mu-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp27-cp27mu-linux_x86_64.whl</a></td>
		<td><a href="http://paddlepaddle.org/download?url=http://paddle-wheel.bj.bcebos.com/latest-gpu-cuda8-cudnn7-avx-mkl/paddlepaddle_gpu-latest-cp27-cp27m-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp27-cp27m-linux_x86_64.whl</a></td>
		<td><a href="http://paddlepaddle.org/download?url=http://paddle-wheel.bj.bcebos.com/latest-gpu-cuda8-cudnn7-avx-mkl/paddlepaddle_gpu-latest-cp35-cp35m-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp35-cp35m-linux_x86_64.whl</a></td>
		<td> <a href="http://paddlepaddle.org/download?url=http://paddle-wheel.bj.bcebos.com/latest-gpu-cuda8-cudnn7-avx-mkl/paddlepaddle_gpu-latest-cp36-cp36m-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp36-cp36m-linux_x86_64.whl</a></td>
		<td> <a href="http://paddlepaddle.org/download?url=http://paddle-wheel.bj.bcebos.com/latest-gpu-cuda8-cudnn7-avx-mkl/paddlepaddle_gpu-latest-cp37-cp37m-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp37-cp37m-linux_x86_64.whl</a></td>
	</tr>
	<tr>
		<td> cuda9.0_cudnn7_avx_mkl </td>
		<td> <a href="http://paddlepaddle.org/download?url=http://paddle-wheel.bj.bcebos.com/latest-gpu-cuda9-cudnn7-avx-mkl/paddlepaddle_gpu-latest-cp27-cp27mu-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp27-cp27mu-linux_x86_64.whl</a></td>
		<td><a href="http://paddlepaddle.org/download?url=http://paddle-wheel.bj.bcebos.com/latest-gpu-cuda9-cudnn7-avx-mkl/paddlepaddle_gpu-latest-cp27-cp27m-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp27-cp27m-linux_x86_64.whl</a></td>
		<td><a href="http://paddlepaddle.org/download?url=http://paddle-wheel.bj.bcebos.com/latest-gpu-cuda9-cudnn7-avx-mkl/paddlepaddle_gpu-latest-cp35-cp35m-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp35-cp35m-linux_x86_64.whl</a></td>
		<td> <a href="http://paddlepaddle.org/download?url=http://paddle-wheel.bj.bcebos.com/latest-gpu-cuda9-cudnn7-avx-mkl/paddlepaddle_gpu-latest-cp36-cp36m-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp36-cp36m-linux_x86_64.whl</a></td>
		<td> <a href="http://paddlepaddle.org/download?url=http://paddle-wheel.bj.bcebos.com/latest-gpu-cuda9-cudnn7-avx-mkl/paddlepaddle_gpu-latest-cp37-cp37m-linux_x86_64.whl"> paddlepaddle_gpu-latest-cp37-cp37m-linux_x86_64.whl</a></td>
	</tr>
   </tbody>
</table>
</p>


<!--TODO this part should be in a new webpage-->

</br></br>

## 在Docker中执行PaddlePaddle训练程序

***

假设您已经在当前目录（比如在/home/work）编写了一个PaddlePaddle的程序: `train.py` （可以参考
[PaddlePaddleBook](http://www.paddlepaddle.org/docs/develop/book/01.fit_a_line/index.cn.html)
编写），就可以使用下面的命令开始执行训练：

     cd /home/work
     docker run -it -v $PWD:/work hub.baidubce.com/paddlepaddle/paddle /work/train.py

上述命令中，`-it` 参数说明容器已交互式运行；`-v $PWD:/work`
指定将当前路径（Linux中PWD变量会展开为当前路径的绝对路径）挂载到容器内部的:`/work`
目录: `hub.baidubce.com/paddlepaddle/paddle` 指定需要使用的容器； 最后`/work/train.py`为容器内执行的命令，即运行训练程序。

当然，您也可以进入到Docker容器中，以交互式的方式执行或调试您的代码：

     docker run -it -v $PWD:/work hub.baidubce.com/paddlepaddle/paddle /bin/bash
     cd /work
     python train.py

**注：PaddlePaddle Docker镜像为了减小体积，默认没有安装vim，您可以在容器中执行** `apt-get install -y vim` **安装后，在容器中编辑代码。**

</br></br>

## 使用Docker启动PaddlePaddle Book教程

***

使用Docker可以快速在本地启动一个包含了PaddlePaddle官方Book教程的Jupyter Notebook，可以通过网页浏览。
PaddlePaddle Book是为用户和开发者制作的一个交互式的Jupyter Notebook。
如果您想要更深入了解deep learning，PaddlePaddle Book一定是您最好的选择。
大家可以通过它阅读教程，或者制作和分享带有代码、公式、图表、文字的交互式文档。

我们提供可以直接运行PaddlePaddle Book的Docker镜像，直接运行：

`docker run -p 8888:8888 hub.baidubce.com/paddlepaddle/book`

国内用户可以使用下面的镜像源来加速访问：

`docker run -p 8888:8888 hub.baidubce.com/paddlepaddle/book`

然后在浏览器中输入以下网址：

`http://localhost:8888/`

就这么简单，享受您的旅程！如有其他问题请参见[FAQ](#FAQ)

</br></br>
## 使用Docker执行GPU训练

***

为了保证GPU驱动能够在镜像里面正常运行，我们推荐使用
[nvidia-docker](https://github.com/NVIDIA/nvidia-docker)来运行镜像。
请不要忘记提前在物理机上安装GPU最新驱动。

`nvidia-docker run -it -v $PWD:/work hub.baidubce.com/paddlepaddle/paddle:latest-gpu /bin/bash`

**注: 如果没有安装nvidia-docker，可以尝试以下的方法，将CUDA库和Linux设备挂载到Docker容器内：**

     export CUDA_SO="$(\ls /usr/lib64/libcuda* | xargs -I{} echo '-v {}:{}') \
     $(\ls /usr/lib64/libnvidia* | xargs -I{} echo '-v {}:{}')"
     export DEVICES=$(\ls /dev/nvidia* | xargs -I{} echo '--device {}:{}')
     docker run ${CUDA_SO} \
      ${DEVICES} -it hub.baidubce.com/paddlepaddle/paddle:latest-gpu


**关于AVX：**

AVX是一种CPU指令集，可以加速PaddlePaddle的计算。最新的PaddlePaddle Docker镜像默认
是开启AVX编译的，所以，如果您的电脑不支持AVX，需要单独[编译](/build_from_source_cn.html) PaddlePaddle为no-avx版本。

以下指令能检查Linux电脑是否支持AVX：

`if cat /proc/cpuinfo | grep -i avx; then echo Yes; else echo No; fi`

如果输出是No，就需要选择使用no-AVX的镜像
