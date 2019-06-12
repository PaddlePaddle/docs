 # 安装与编译

## Linux安装PaddlePaddle
### Q：Ubuntu18.10、CPU版本、Python3.6编译错误如何解决？
+ 版本、环境信息
1）PaddlePaddle版本：Github develop版本 
2）CPU：vmware14 
3）GPU：非GPU
4）系统环境：Ubuntu18.10 64位 Python3.6
+ 安装方式信息：[本地编译](http://paddlepaddle.org/documentation/docs/zh/1.3/beginners_guide/install/compile/compile_Ubuntu.html/#ubt_source)中本机编译第十步make 命令
+ 错误信息
```
[100%] Built target warpctc
Install the project...
-- Install configuration: "Release"
-- Installing: /home/eason/main/source/ai/alg/Paddle/Paddle/build/third_party/install/warpctc/lib/libwarpctc.so
-- Installing: /home/eason/main/source/ai/alg/Paddle/Paddle/build/third_party/install/warpctc/include/ctc.h
[ 5%] Completed 'extern_warpctc'
[ 5%] Built target extern_warpctc
Scanning dependencies of target extern_gzstream
[ 5%] Creating directories for 'extern_gzstream'
[ 5%] Performing download step (git clone) for 'extern_gzstream'
Cloning into 'extern_gzstream'...
Already on 'master'
Your branch is up to date with 'origin/master'.
[ 5%] No patch step for 'extern_gzstream'
[ 5%] No update step for 'extern_gzstream'
[ 5%] No configure step for 'extern_gzstream'
[ 5%] Performing build step for 'extern_gzstream'
CPPFLAGS: "-I/home/eason/main/source/ai/alg/Paddle/Paddle/build/third_party/install/zlib/include" -I. -fPIC -O
LDFLAGS: "-L/home/eason/main/source/ai/alg/Paddle/Paddle/build/third_party/install/zlib/lib" -L. -lgzstream -lz
[ 5%] Performing install step for 'extern_gzstream'
[ 5%] Completed 'extern_gzstream'
[ 5%] Built target extern_gzstream
Scanning dependencies of target place
[ 5%] Building CXX object paddle/fluid/platform/CMakeFiles/place.dir/place.cc.o
In file included from /home/eason/main/source/ai/alg/Paddle/Paddle/build/third_party/boost/src/extern_boost/boost/mpl/aux_/na_assert.hpp:23,
from /home/eason/main/source/ai/alg/Paddle/Paddle/build/third_party/boost/src/extern_boost/boost/mpl/arg.hpp:25,
from /home/eason/main/source/ai/alg/Paddle/Paddle/build/third_party/boost/src/extern_boost/boost/variant/variant_fwd.hpp:19,
from /home/eason/main/source/ai/alg/Paddle/Paddle/build/third_party/boost/src/extern_boost/boost/variant/variant.hpp:27,
from /home/eason/main/source/ai/alg/Paddle/Paddle/build/third_party/boost/src/extern_boost/boost/variant.hpp:17,
from /home/eason/main/source/ai/alg/Paddle/Paddle/paddle/fluid/platform/variant.h:45,
from /home/eason/main/source/ai/alg/Paddle/Paddle/paddle/fluid/platform/place.h:21,
from /home/eason/main/source/ai/alg/Paddle/Paddle/paddle/fluid/platform/place.cc:15:
/home/eason/main/source/ai/alg/Paddle/Paddle/build/third_party/boost/src/extern_boost/boost/mpl/assert.hpp:154:21: error: unnecessary parentheses in declaration of ‘assert_arg’ [-Werror=parentheses]
failed ************ (Pred::************
^
/home/eason/main/source/ai/alg/Paddle/Paddle/build/third_party/boost/src/extern_boost/boost/mpl/assert.hpp:159:21: error: unnecessary parentheses in declaration of ‘assert_not_arg’ [-Werror=parentheses]
failed ************ (boost::mpl::not_::************
^
cc1plus: all warnings being treated as errors
make[2]: *** [paddle/fluid/platform/CMakeFiles/place.dir/build.make:63: paddle/fluid/platform/CMakeFiles/place.dir/place.cc.o] Error 1
make[1]: *** [CMakeFiles/Makefile2:2670: paddle/fluid/platform/CMakeFiles/place.dir/all] Error 2
make: *** [Makefile:152: all] Error 2
```
+ 问题解答
自行编译建议的GCC版本:4.8、5.4以及更高。

### Q：遇到如下cuDNN报错如何解决？
```
CUDNN_STATUS_NOT_INITIALIZED at [/paddle/paddle/fluid/platform/device_context.cc:216]
```

+ 问题解答
cuDNN与CUDA版本不一致导致。PIP安装的GPU版本默认使用CUDA 9.0和cuDNN 7编译，请根据您的环境配置选择在官网首页选择对应的安装包进行安装，例如paddlepaddle-gpu==1.2.0.post87 代表使用CUDA 8.0和cuDNN 7编译的1.2.0版本。

### Q：cuda9.0需要安装哪一个版本的paddle，安装包在哪?
- 问题解答
pip install paddlepaddle-gpu 命令将安装支持CUDA 9.0 cuDNN v7的PaddlePaddle，可以参考[安装说明文档](http://paddlepaddle.org/documentation/docs/zh/1.4/beginners_guide/install/index_cn.html)


### Q：使用  `pip install paddlepaddle-gpu==0.14.0.post87`命令在公司内部开发GPU机器上安装PaddlePaddle，安装信息如下：
![](https://user-images.githubusercontent.com/12878507/45028894-606ba980-b079-11e8-98e7-6e80f1c3f386.png)
机器的CUDA信息如下：
![](https://user-images.githubusercontent.com/12878507/45028950-8c872a80-b079-11e8-82f2-ca6591203eb1.png)
按照官网安装：pip install paddlepaddle-gpu==0.14.0.post87
执行 import paddle.fluid as fluid 失败
![](https://user-images.githubusercontent.com/12878507/45028976-a0329100-b079-11e8-84a7-07253eafb3cb.png)
奇怪的是，同样的环境下，上周运行成功，这周确运行失败，求解答？

+ 问题解答
这通常是GPU显存不足导致的，请检查一下机器的显存，确保显存足够后再尝试import paddle.fluid


### Q：在使用PaddlePaddle GPU的Docker镜像的时候，出现 `Cuda Error: CUDA driver version is insufficient for CUDA runtime version`？
+ 问题解答
通常出现 `Cuda Error: CUDA driver version is insufficient for CUDA runtime version`, 原因在于没有把机器上CUDA相关的驱动和库映射到容器内部。
使用nvidia-docker, 命令只需要将docker换为nvidia-docker即可。
更多请参考[nvidia-docker](https://github.com/NVIDIA/nvidia-docker)


### Q：安成功安装了PaddlePaddle CPU版本后，使用Paddle训练模型，训练过程中，Paddle会自动退出，gdb显示Illegal instruction？
+ 报错信息
```bash
*** Aborted at 1539697466 (unix time) try "date -d @1539697466" if you are using GNU date ***
PC: @                0x0 (unknown)
*** SIGILL (@0x7fe3a27b7912) received by PID 13005 (TID 0x7fe4059d8700) from PID 18446744072140585234; stack trace: ***
    @       0x318b20f500 (unknown)
    @     0x7fe3a27b7912 paddle::framework::VisitDataType<>()
    @     0x7fe3a279f84f paddle::operators::math::set_constant_with_place<>()
    @     0x7fe3a1e50c21 paddle::operators::FillConstantOp::RunImpl()
    @     0x7fe3a27526bf paddle::framework::OperatorBase::Run()
    @     0x7fe3a1ca31ea paddle::framework::Executor::RunPreparedContext()
    @     0x7fe3a1ca3be0 paddle::framework::Executor::Run()
    @     0x7fe3a1bc9e7d _ZZN8pybind1112cpp_function10initializeIZN6paddle6pybindL13pybind11_initEvEUlRNS2_9framework8ExecutorERKNS4_11ProgramDescEPNS4_5ScopeEibbE63_vIS6_S9_SB_ibbEINS_4nameENS_9is_methodENS_7siblingEEEEvOT_PFT0_DpT1_EDpRKT2_ENUlRNS_6detail13function_callEE1_4_FUNEST_
    @     0x7fe3a1c14c24 pybind11::cpp_function::dispatcher()
    @     0x7fe405acf3e4 PyEval_EvalFrameEx
    @     0x7fe405ad0130 PyEval_EvalCodeEx
    @     0x7fe405ace4a1 PyEval_EvalFrameEx
    @     0x7fe405ad0130 PyEval_EvalCodeEx
    @     0x7fe405ace4a1 PyEval_EvalFrameEx
    @     0x7fe405ad0130 PyEval_EvalCodeEx
    @     0x7fe405a5c181 function_call
    @     0x7fe405a340f3 PyObject_Call
    @     0x7fe405accde7 PyEval_EvalFrameEx
    @     0x7fe405acec56 PyEval_EvalFrameEx
    @     0x7fe405ad0130 PyEval_EvalCodeEx
    @     0x7fe405a5c27d function_call
    @     0x7fe405a340f3 PyObject_Call
    @     0x7fe405accde7 PyEval_EvalFrameEx
    @     0x7fe405ad0130 PyEval_EvalCodeEx
    @     0x7fe405a5c181 function_call
    @     0x7fe405a340f3 PyObject_Call
    @     0x7fe405a46f7f instancemethod_call
    @     0x7fe405a340f3 PyObject_Call
    @     0x7fe405a8abd4 slot_tp_call
    @     0x7fe405a340f3 PyObject_Call
    @     0x7fe405acd887 PyEval_EvalFrameEx
    @     0x7fe405acec56 PyEval_EvalFrameEx
```
+ 问题解答
CPU版本PaddlePaddle自动退出的原因通常是因为所在机器不支持AVX2指令集而主动abort。简单的判断方法：
用gdb-7.9以上版本（因编译C++文件用的工具集是gcc-4.8.2，目前只知道gdb-7.9这个版本可以debug gcc4编译出来的目标文件）：
```bash
$ /path/to/gdb -iex "set auto-load safe-path /" -iex "set solib-search-path /path/to/gcc-4/lib" /path/to/python -c core.xxx
```

在gdb界面：
```bash
(gdb) disas
```

找到箭头所指的指令，例如：
```bash
   0x00007f381ae4b90d <+3101>:  test   %r8,%r8
=> 0x00007f381ae4b912 <+3106>:  vbroadcastss %xmm0,%ymm1
   0x00007f381ae4b917 <+3111>:  lea    (%r12,%rdx,4),%rdi
```

然后google一下这个指令需要的指令集。上面例子中的带xmm和ymm操作数的vbroadcastss指令只在AVX2中支持
然后看下自己的CPU是否支持该指令集
```bash
cat /proc/cpuinfo | grep flags | uniq | grep avx --color
```

如果没有AVX2，就表示确实是指令集不支持引起的主动abort
如果没有AVX2指令集，就需要要安装不支持AVX2指令集版本的PaddlePaddle，默认安装的PaddlePaddle是支持AVX2指令集的，因为AVX2可以加速模型训练的过程，更多细节可以参考[安装文档](http://paddlepaddle.org/documentation/docs/zh/1.4/beginners_guide/install/index_cn.html)

### Q：使用`sudo nvidia-docker run --name Paddle -it -v $PWD:/work hub.baidubce.com/paddlepaddle/paddle:latest-gpu-cuda8.0-cudnn7 /bin/bash`，安装成功后，出现如下问题
```bash
import paddle.fluid
*** Aborted at 1539682149 (unix time) try "date -d @1539682149" if you are using GNU date ***
PC: @ 0x0 (unknown)
*** SIGILL (@0x7f6ac6ea9436) received by PID 16 (TID 0x7f6b07bc7700) from PID 18446744072751846454; stack trace: ***
```

+ 问题解答
请先确定一下机器是否支持AVX2指令集，如果不支持，请按照相应的不支持AVX2指令集的PaddlePaddle，可以解决该问题。


### Q：使用的系统是Ubuntu 16.04，GPU相关环境：cuda8.0, cudnn 6.0, 安装最新版的paddlepaddle fluid 后，`import paddle.fluid`时问题如下：
+ 报错信息
```bash
Traceback (most recent call last):
File "", line 1, in 
File "/usr/local/lib/python2.7/dist-packages/paddle/fluid/init.py", line 132, in 
bootstrap()
File "/usr/local/lib/python2.7/dist-packages/paddle/fluid/init.py", line 126, in bootstrap
core.init_devices(not in_test)
paddle.fluid.core.EnforceNotMet: CUBLAS: not initialized, at [/paddle/paddle/fluid/platform/device_context.cc:153]
PaddlePaddle Call Stacks:
0 0x7f0238da06f6p paddle::platform::EnforceNotMet::EnforceNotMet(std::exception_ptr::exception_ptr, char const*, int) + 486
1 0x7f0239b1ee54p paddle::platform::CUDADeviceContext::CUDADeviceContext(paddle::platform::CUDAPlace) + 1684
2 0x7f0239b1feb0p paddle::platform::DeviceContextPool::DeviceContextPool(std::vector<boost::variant<paddle::platform::CUDAPlace, paddle::platform::CPUPlace, paddle::platform::CUDAPinnedPlace, boost::detail::variant::void, boost::detail::variant::void, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_>, std::allocator<boost::variant<paddle::platform::CUDAPlace, paddle::platform::CPUPlace, paddle::platform::CUDAPinnedPlace, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_> > > const&) + 752
3 0x7f0238e368bcp paddle::framework::InitDevices(bool, std::vector<int, std::allocator >) + 588
4 0x7f0238e36addp paddle::framework::InitDevices(bool) + 285
5 0x7f0238d865bap
6 0x7f0238db1804p pybind11::cpp_function::dispatcher(_object*, _object*, _object*) + 2596
7 0x4bc3fap PyEval_EvalFrameEx + 1482
8 0x4b9ab6p PyEval_EvalCodeEx + 774
9 0x4c1e6fp PyEval_EvalFrameEx + 24639
10 0x4b9ab6p PyEval_EvalCodeEx + 774
11 0x4b97a6p PyEval_EvalCode + 22
12 0x4b96dfp PyImport_ExecCodeModuleEx + 191
13 0x4b2b06p
14 0x4b402cp
15 0x4a4ae1p
16 0x4a4513p PyImport_ImportModuleLevel + 2259
17 0x4a59e4p
18 0x4a577ep PyObject_Call + 62
19 0x4c5e10p PyEval_CallObjectWithKeywords + 48
20 0x4be6d7p PyEval_EvalFrameEx + 10407
21 0x4b9ab6p PyEval_EvalCodeEx + 774
22 0x4eb30fp
23 0x44a7a2p PyRun_InteractiveOneFlags + 400
24 0x44a56dp PyRun_InteractiveLoopFlags + 186
25 0x43092ep
26 0x493ae2p Py_Main + 1554
27 0x7f026bfa1830p __libc_start_main + 240
28 0x4933e9p _start + 41
```
+ 问题解答
请先查看您系统GPU环境的适配关系，应该选择和您的系统已经安装的CUDA版本相同的whl包，您的系统是cuda 8.0, cudnn 6 应该使用cuda8.0_cudnn7_avx_mkl才可以适配。

然后尝试`import paddle.fluid`命令看看是否报错。
如果报错，则可能是GPU 和CUDA环境没有正确配置。
如果没有报错，请判断是否有给所有相关文件`sudo权限`。



### Q：安装的是cuda9.0和cudnn7.0，默认安装的是0.14.0.post87，训练一个手写数据那个例子的时候报错？
+ 报错信息：
```bash
Traceback (most recent call last):
  File "train.py", line 240, in <module>
    main()
  File "train.py", line 236, in main
    train(args)
  File "train.py", line 147, in train
    exe.run(fluid.default_startup_program())
  File "/usr/local/lib/python2.7/dist-packages/paddle/fluid/executor.py", line 443, in run
    self.executor.run(program.desc, scope, 0, True, True)
paddle.fluid.core.EnforceNotMet: enforce allocating <= available failed, 1827927622 > 1359806208
 at [/paddle/paddle/fluid/platform/gpu_info.cc:119]
PaddlePaddle Call Stacks: 
0       0x7f1bac5312f6p paddle::platform::EnforceNotMet::EnforceNotMet(std::__exception_ptr::exception_ptr, char const*, int) + 486
1       0x7f1bad3a95bep paddle::platform::GpuMaxChunkSize() + 766
2       0x7f1bad2d92ddp paddle::memory::GetGPUBuddyAllocator(int) + 141
3       0x7f1bad2d94ecp void* paddle::memory::Alloc<paddle::platform::CUDAPlace>(paddle::platform::CUDAPlace, unsigned long) + 28
4       0x7f1bad2ced42p paddle::framework::Tensor::mutable_data(boost::variant<paddle::platform::CUDAPlace, paddle::platform::CPUPlace, paddle::platform::CUDAPinnedPlace, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_>, std::type_index) + 866
5       0x7f1bac7a0cbfp paddle::operators::FillConstantOp::RunImpl(paddle::framework::Scope const&, boost::variant<paddle::platform::CUDAPlace, paddle::platform::CPUPlace, paddle::platform::CUDAPinnedPlace, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_> const&) const + 1007
6       0x7f1bad261ebdp paddle::framework::OperatorBase::Run(paddle::framework::Scope const&, boost::variant<paddle::platform::CUDAPlace, paddle::platform::CPUPlace, paddle::platform::CUDAPinnedPlace, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_, boost::detail::variant::void_> const&) + 205
7       0x7f1bac5cd06fp paddle::framework::Executor::RunPreparedContext(paddle::framework::ExecutorPrepareContext*, paddle::framework::Scope*, bool, bool, bool) + 255
8       0x7f1bac5ce0c0p paddle::framework::Executor::Run(paddle::framework::ProgramDesc const&, paddle::framework::Scope*, int, bool, bool) + 128
9       0x7f1bac548cbbp void pybind11::cpp_function::initialize<pybind11::cpp_function::initialize<void, paddle::framework::Executor, paddle::framework::ProgramDesc const&, paddle::framework::Scope*, int, bool, bool, pybind11::name, pybind11::is_method, pybind11::sibling>(void (paddle::framework::Executor::*)(paddle::framework::ProgramDesc const&, paddle::framework::Scope*, int, bool, bool), pybind11::name const&, pybind11::is_method const&, pybind11::sibling const&)::{lambda(paddle::framework::Executor*, paddle::framework::ProgramDesc const&, paddle::framework::Scope*, int, bool, bool)#1}, void, paddle::framework::Executor*, paddle::framework::ProgramDesc const&, paddle::framework::Scope*, int, bool, bool, pybind11::name, pybind11::is_method, pybind11::sibling>(pybind11::cpp_function::initialize<void, paddle::framework::Executor, paddle::framework::ProgramDesc const&, paddle::framework::Scope*, int, bool, bool, pybind11::name, pybind11::is_method, pybind11::sibling>(void (paddle::framework::Executor::*)(paddle::framework::ProgramDesc const&, paddle::framework::Scope*, int, bool, bool), pybind11::name const&, pybind11::is_method const&, pybind11::sibling const&)::{lambda(paddle::framework::Executor*, paddle::framework::ProgramDesc const&, paddle::framework::Scope*, int, bool, bool)#1}&&, void (*)(paddle::framework::Executor*, paddle::framework::ProgramDesc const&, paddle::framework::Scope*, int, bool, bool), pybind11::name const&, pybind11::is_method const&, pybind11::sibling const&)::{lambda(pybind11::detail::function_call&)#3}::_FUN(pybind11::detail::function_call) + 555
10      0x7f1bac5411c4p pybind11::cpp_function::dispatcher(_object*, _object*, _object*) + 2596
11            0x4c37edp PyEval_EvalFrameEx + 31165
12            0x4b9ab6p PyEval_EvalCodeEx + 774
13            0x4c16e7p PyEval_EvalFrameEx + 22711
14            0x4b9ab6p PyEval_EvalCodeEx + 774
15            0x4c1e6fp PyEval_EvalFrameEx + 24639
16            0x4c136fp PyEval_EvalFrameEx + 21823
17            0x4b9ab6p PyEval_EvalCodeEx + 774
18            0x4eb30fp
19            0x4e5422p PyRun_FileExFlags + 130
20            0x4e3cd6p PyRun_SimpleFileExFlags + 390
21            0x493ae2p Py_Main + 1554
22      0x7f1bd6ae9830p __libc_start_main + 240
23            0x4933e9p _start + 41
```
+ 问题解答
该问题通常是GPU显存不足造成的，请在显存充足的GPU服务器上再次尝试则可。可以检查一下机器的显存使用情况。
方法如下：
```bash
test@test:~$ nvidia-smi
Tue Jul 24 08:24:22 2018       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 384.130                Driver Version: 384.130                   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX 960     Off  | 00000000:01:00.0  On |                  N/A |
| 22%   52C    P2   100W / 120W |   1757MiB /  1994MiB |     98%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    0      1071      G   /usr/lib/xorg/Xorg                           314MiB |
|    0      1622      G   compiz                                       149MiB |
|    0      2201      G   fcitx-qimpanel                                 7MiB |
|    0     15304      G   ...-token=58D78B2D4A63DAE7ED838021B2136723    74MiB |
|    0     15598      C   python                                      1197MiB |
+-----------------------------------------------------------------------------+
```

### Q：版本为paddlepaddle_gpu-0.14.0.post87-cp27-cp27mu-manylinux1_x86_64.whl，跑一个简单的测试程序，出现Segmentation fault。其中 如果place为cpu，可以正常输出，改成gpu则core。
+ 程序代码
```bash
def testpaddle014():
    place = fluid.CUDAPlace(0)
    #place = fluid.CPUPlace()
    print 'version', paddle.__version__, place
    input = fluid.layers.data(name='input', shape=[3,50,50], dtype='float32')
    
    output = fluid.layers.conv2d(input=input,num_filters=1,filter_size=3,stride=1,padding=1,groups=1,act=None)
    #output = fluid.layers.fc(input=input,size=2)
    
    fetch_list = [output.name]
    data = np.zeros((2,3,50,50), np.float32)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    outputlist = exe.run(
                fluid.default_main_program(),
                feed={'input': data},
                fetch_list=fetch_list
            )
    print 'output', outputlist[0].shape
```
+ 问题解答
安装版本为`paddlepaddle_gpu-0.14.0.post87-cp27-cp27mu-manylinux1_x86_64.whl`，其中post87是指在CUDA8.0、cudnn7.0编译的，请确定您机器上是否安装了对应版本的cuDNN。造成问题描述中现象的情况通常可能是环境不匹配导致的。

### Q：安装完了PaddlePaddle后，出现以下python相关的单元测试都过不了的情况：
```
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
```
并且查询PaddlePaddle单元测试的日志，提示：
```
    paddle package is already in your PYTHONPATH. But unittest need a clean environment.
    Please uninstall paddle package before start unittest. Try to 'pip uninstall paddle'.
```

+ 问题解答
卸载PaddlePaddle包 `pip uninstall paddle`, 清理掉老旧的PaddlePaddle安装包，使得单元测试有一个干净的环境。如果PaddlePaddle包已经在python的site-packages里面，单元测试会引用site-packages里面的python包，而不是源码目录里 `/python` 目录下的python包。同时，即便设置 `PYTHONPATH` 到 `/python` 也没用，因为python的搜索路径是优先已经安装的python包。

### Q：根据官方文档中提供的步骤安装Docker，无法下载需要的golang，导致`tar: Error is not recoverable: exiting now`？
+ 报错截图
![](https://user-images.githubusercontent.com/17102274/42516245-314346be-8490-11e8-85cc-eb95e9f0e02c.png)

+ 问题解答
由上图可知，生成docker镜像时需要下载[golang](https://storage.googleapis.com/golang/go1.8.1.linux-amd64.tar.gz)，使用者需要保证电脑可以科学上网。
选择下载并使用docker.paddlepaddlehub.com/paddle:latest-devdocker镜像，执行命令如下：
```
git clone https://github.com/PaddlePaddle/Paddle.git

cd Paddle

git checkout -b 0.14.0 origin/release/0.14.0


sudo docker run --name paddle-test -v $PWD:/paddle --network=host -it docker.paddlepaddlehub.com/paddle:latest-dev /bin/bash
```
进入docker编译GPU版本的PaddlePaddle，执行命令如下：
```
mkdir build && cd build
# 编译GPU版本的PaddlePaddle
cmake .. -DWITH_FLUID_ONLY=ON -DWITH_GPU=ON -DWITH_TESTING=ON
make -j$(nproc)
```
通过上面的方式操作后：
![](https://user-images.githubusercontent.com/17102274/42516287-46ccae8a-8490-11e8-9186-985efff3629c.png)
接着安装PaddlePaddle并运行线性回归test_fit_a_line.py程序测试一下PaddlePaddle是安装成功则可
```bash
pip install build/python/dist/*.whl
python python/paddle/fluid/tests/book/test_fit_a_line.py
```

### Q：在Docker镜像上，GPU版本的PaddlePaddle运行结果报错
![](https://user-images.githubusercontent.com/17102274/42516300-50f04f8e-8490-11e8-95f1-613d3d3f6ca6.png)
![](https://user-images.githubusercontent.com/17102274/42516303-5594bd22-8490-11e8-8c01-55741484f126.png)

+ 问题解答
使用`sudo docker run --name paddle-test -v $PWD:/paddle --network=host -it docker.paddlepaddlehub.com/paddle:latest-dev /bin/bash`命令创建的docker容器仅能支持运行CPU版本的PaddlePaddle。
使用如下命令重新开启支持GPU运行的docker容器：
```
export CUDA_SO="$(\ls /usr/lib64/libcuda* | xargs -I{} echo '-v {}:{}') $(\ls /usr/lib64/libnvidia* | xargs -I{} echo '-v {}:{}')"

export DEVICES=$(\ls /dev/nvidia* | xargs -I{} echo '--device {}:{}')

sudo docker run ${CUDA_SO} ${DEVICES} --rm --name paddle-test-gpu -v /usr/bin/nvidia-smi:/usr/bin/nvidia-smi -v $PWD:/paddle --network=host -it docker.paddlepaddlehub.com/paddle:latest-dev /bin/bash
```
进入docker之后执行如下命令进行PaddlePaddle的安装及测试运行：
```
export LD_LIBRARY_PATH=/usr/lib64:/usr/local/lib:$LD_LIBRARY_PATH
pip install build/python/dist/*.whl
python python/paddle/fluid/tests/book/test_fit_a_line.py
```

### Q：在Liunx环境上，通过编译源码的方式安装PaddlePaddle，当安装成功后，运行 `paddle version`, 出现 `PaddlePaddle 0.0.0`？
+ 问题解答
如果运行 `paddle version`, 出现`PaddlePaddle 0.0.0`；或者运行 `cmake ..`，出现
```bash
CMake Warning at cmake/version.cmake:20 (message):
Cannot add paddle version from git tag
```
在dev分支下这个情况是正常的，在release分支下通过export PADDLE_VERSION=对应版本号 来解决。


### Q：安装PaddlePaddle过程中，出现`paddlepaddle\*.whl is not a supported wheel on this platform`？
+ 问题解答
`paddlepaddle\*.whl is not a supported wheel on this platform`表示你当前使用的PaddlePaddle不支持你当前使用的系统平台，即没有找到和当前系统匹配的paddlepaddle安装包。最新的paddlepaddle python安装包支持Linux x86_64和MacOS 10.12操作系统，并安装了python 2.7和pip 9.0.1。
请先尝试安装最新的pip，方法如下：
```bash
pip install --upgrade pip
```
如果还不行，可以执行 `python -c "import pip; print(pip.pep425tags.get_supported())"` 获取当前系统支持的python包的后缀，
并对比是否和正在安装的后缀一致。
如果系统支持的是 `linux_x86_64` 而安装包是 `manylinux1_x86_64` ，需要升级pip版本到最新；
如果系统支持 `manylinux1_x86_64` 而安装包（本地）是 `linux_x86_64` ，可以重命名这个whl包为 `manylinux1_x86_64` 再安装。


## MacOS安装PaddlePaddle

### Q：PaddlePaddle官方文档中，关于MacOS下安装PaddlePaddle只提及了MacOS中使用Docker环境安装PaddlePaddle的内容，没有Mac本机安装的内容？
+ 问题解答
基于Docker容器编译PaddlePaddle与本机上直接编译PaddlePaddle，所使用的编译执行命令是不一样的，但是官网仅仅给出了基于Docker容器编译PaddlePaddle所执行的命令。
	1.基于Docker容器编译PaddlePaddle，需要执行：
	```bash
	# 1. 获取源码

	git clone https://github.com/PaddlePaddle/Paddle.git

	cd Paddle

	# 2. 可选步骤：源码中构建用于编译PaddlePaddle的Docker镜像

	docker build -t paddle:dev .

	# 3. 执行下面的命令编译CPU-Only的二进制

	docker run -it -v $PWD:/paddle -e "WITH_GPU=OFF" -e "WITH_TESTING=OFF" paddlepaddle/paddle_manylinux_devel:cuda8.0_cudnn5 bash -x /paddle/paddle/scripts/paddle_build.sh build

	# 4. 或者也可以使用为上述可选步骤构建的镜像（必须先执行第2步）

	docker run -it -v $PWD:/paddle -e "WITH_GPU=OFF" -e "WITH_TESTING=OFF" paddle:dev
	```

	2.直接在本机上编译PaddlePaddle，需要执行：
	```bash
	# 1. 使用virtualenvwrapper创建python虚环境并将工作空间切换到虚环境

	mkvirtualenv paddle-venv

	workon paddle-venv

	# 2. 获取源码

	git clone https://github.com/PaddlePaddle/Paddle.git

	cd Paddle

	# 3. 执行下面的命令编译CPU-Only的二进制

	mkdir build && cd build

	cmake .. -DWITH_GPU=OFF -DWITH_TESTING=OFF

	make -j$(nproc)
	```
更详细的内容，请参考[官方文档](http://paddlepaddle.org/documentation/docs/zh/1.4/beginners_guide/install/install_MacOS.html)

### Q：以源码方式在MacOS上安装时，出现`Configuring incomplete, errors occured!`？
+ 报错截图
![](https://user-images.githubusercontent.com/17102274/42515239-e24be824-848d-11e8-9f3d-3baf156dcea8.png)
![](https://user-images.githubusercontent.com/17102274/42515246-e6f7c2d0-848d-11e8-853a-7d7401e4650f.png)
+ 问题解答
	安装PaddlePaddle编译时需要的各种依赖则可，如下：

	```bash
	pip install wheel
	brew install protobuf@3.1
	pip install protobuf==3.1.0
	```

	如果执行pip install protobuf==3.1.0时报错，输出下图内容：

	![](https://user-images.githubusercontent.com/17102274/42515286-fb7a7b76-848d-11e8-931a-a7f61bd6374b.png)

	从图中可以获得报错的关键为`Cannot uninstall 'six'`，那么解决方法就是先安装好`six`，再尝试安装`protobuf 3.1.0`如下：

	```bash
	easy_install -U six 
	pip install protobuf==3.1.0
	```

### Q：MacOS 10.12下编译PaddlePaddle出现`/bin/sh: wget: command not found`，如何解决？
![](https://user-images.githubusercontent.com/17102274/42515304-0bd7012e-848e-11e8-966f-946361ac7a56.png)

+ 问题解答
报错的原因从报错输出的信息中可以发现，即没有有找到wget命令，安装wget则可，安装命令如下：
```bash
brew install wget
```

### Q：官网中只介绍了Mac下使用Docker安装编译PaddlePaddle的方式，因为我对Docker不怎么熟悉，想直接安装到本地的Mac系统中，MacOS版本为10.13，是符合要求的，但尝试了多次后，已经出现`No rule to make target`错误？
+ 报错截图
![](https://user-images.githubusercontent.com/17102274/42515324-1bd9c020-848e-11e8-8934-d7da5fc1f090.png)

+ 问题解答
该问题是有CMake引擎的，修改CMake编译命令，打开WITH_FLUID_ONLY编译选项，修改后编译命令如下：
	```bash
	cmake .. -DWITH_FLUID_ONLY=ON -DWITH_GPU=OFF -DWITH_TESTING=OFF
	```

### Q：MacOS本机直接通过源码编译的方式安装PaddlePaddle出现`[paddle/fluid/platform/CMakeFiles/profiler_py_proto.dir/all] Error 2`？
+ 报错截图
![](https://user-images.githubusercontent.com/17102274/42515350-28c055ce-848e-11e8-9b90-c294b375d8a4.png)
+ 问题解答
    使用cmake版本为3.4则可


### Q：MacOS本地编译PaddlePaddle github上develop分支的代码出现，出现No such file or directory错误？
+ 报错截图
![](https://user-images.githubusercontent.com/17102274/42515402-453cc0d4-848e-11e8-9a03-a579ea8e4d2d.png)

+ 问题解答
因为此时develop分支上Generating build/.timestamp这一步涉及的代码还在进行修改，所以并不能保证稳定，建议切换回稳定分支进行编译安装。
	可以通过执行如下命令将分支切换到0.14.0进行编译:
	```bash
	cd Paddle
	git checkout -b release/1.1
	cd build &&  rm -rf *
	cmake .. -DWITH_FLUID_ONLY=ON -DWITH_GPU=OFF -DWITH_TESTING=OFF
	make -j4
	```
	编译成功后的结果如图：
	![](https://user-images.githubusercontent.com/17102274/42515418-4fb71e56-848e-11e8-81c6-da2a5553a27a.png)



### Q：paddle源码编译（osx）报各种module找不到的问题
从源码编译，最后`cmake ..`时
`Could NOT find PY_google.protobuf (missing: PY_GOOGLE.PROTOBUF)
CMake Error at cmake/FindPythonModule.cmake:27 (message):
python module google.protobuf is not found`
若通过-D设置路径后，又会有其他的如`Could not find PY_wheel`等其他找不到的情况
+ 问题解答
![](https://cloud.githubusercontent.com/assets/728699/19915727/51f7cb68-a0ef-11e6-86cc-febf82a07602.png)
如上，当cmake找到python解释器和python库时，如果安装了许多pythons，它总会找到不同版本的Python。在这种情况下，您应该明确选择应该使用哪个python。
通过cmake显式设置python包。只要确保python libs和python解释器是相同的python可以解决所有这些问题。当这个python包有一些原生扩展时，例如numpy，显式set python包可能会失败。

### Q：在MacOS下，本地直接编译安装PaddlePaddle遇到`collect2: ld terminated with signal 9 [Killed] ` ？
+ 问题解答
该问题是由磁盘空间不足造成的，你的硬盘要有30G+的空余空间，请尝试清理出足够的磁盘空间，重新安装。


### Q：因为需要安装numpy等包，但在Mac自带的Python上无法安装，权限错误导致难以将PaddlePaddle正常安装到Mac本地？
+ 问题解答
Mac上对自带的Python和包有严格的权限保护，最好不要在自带的Python上安装。建议用virtualenv建立一个新的Python环境来操作。

virtualenv的基本原理是将机器上的Python运行所需的运行环境完整地拷贝一份。我们可以在一台机器上制造多份拷贝，并在这多个拷贝之间自由切换，这样就相当于在一台机器上拥有了多个相互隔离、互不干扰的Python环境。

下面使用virtualenv为Paddle生成一个专用的Python环境。
	安装virtualenv，virtualenv本身也是Python的一个包，可以用pip进行安装：
	```
	 sudo -H pip install virtualenv
	```

	由于virtualenv需要安装给系统自带的Python，因此需要使用sudo权限。接着使用安装好的virtualenv创建一个新的Python运行环境：
	```
	virtualenv --no-site-packages paddle
	```

	--no-site-packages 参数表示不拷贝已有的任何第三方包，创造一个完全干净的新Python环境。后面的paddle是我们为这个新创建的环境取的名字。执行完这一步后，当前目录下应该会出现一个名为paddle（或者你取的其他名字）的目录。这个目录里保存了运行一个Python环境所需要的各种文件。

	启动运行环境：
	```
	source paddle/bin/activate
	```

	执行后会发现命令提示符前面增加了(paddle)字样，说明已经成功启动了名为‘paddle’的Python环境。执行which python，可以发现使用的已经是刚刚创建的paddle目录下的Python。
	在这个环境中，我们可以自由地进行PaddlePaddle的安装、使用和开发工作，无需担心对系统自带Python的影响。
	如果我们经常使用Paddle这个环境，我们每次打开终端后都需要执行一下source paddle/bin/activate来启动环境，比较繁琐。为了简便，可以修改终端的配置文件，来让终端每次启动后自动启动特定的Python环境。
	执行:

	```
	vi ~/.bash_profile
	```

	打开终端配置文件，并在文件的最后添加一行：
	```
	source paddle/bin/activate
	```
	这样，每次打开终端时就会自动启动名为‘paddle’的Python环境了。
