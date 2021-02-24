.. _cn_api_paddle_utils_cpp_extension_load:

load
-------------------------------

.. py:function:: paddle.utils.cpp_extension.load(name, sources, extra_cxx_cflags=None, extra_cuda_cflags=None, extra_ldflags=None, extra_include_paths=None, build_directory=None, interpreter=None, verbose=False)

此接口将即时编译（Just-In-Time）传入的自定义 OP 对应的 cpp 和 cuda 源码文件，返回一个包含自定义算子 API 的 ``Module`` 对象。

其通过子进程的方式，在后台隐式地执行源码文件编译、符号链接、动态库生成、组网 API 接口生成等一系列过程。不需要本地预装 CMake 或者 Ninja 等工具命令，仅需必要的编译器命令环境，如 Linux 下需安装版本不低于 5.4 的 GCC，并软链到 ``/usr/bin/cc`` ；若编译支持 GPU 设备的算子，则需要预装 ``nvcc`` 编译环境。

同时，编译前会执行 ABI 兼容性检查，即检查 ``cc`` 命令对应的 GCC 版本是否与编译本地安装的 Paddle 时的 GCC 版本一致。如对于 CUDA 10.1 以上的 Paddle 默认使用 GCC 8.2 编译，则本地 ``cc`` 对应的编译器版本也需为 8.2 ，否则可能由于 ABI 兼容性原因引发自定义 OP 执行期报错。

相对于 :ref:`cn_api_paddle_utils_cpp_extension_setup` 的方式，此接口不需要额外的 ``setup.py`` 文件和  ``python setup.py install`` 命令，``load``  接口包含了一键执行自定义 OP 的编译和加载的全部流程。

.. note::

    1. 编译器的 ABI 兼容性是向前兼容的，Linux 下推荐使用 GCC 8.2 高版本作为 ``/usr/bin/cc`` 命令的软链对象。
    2. Linux 下可通过 ``which cc`` 查看 ``cc`` 命令的位置；使用 ``cc --version`` 查看对应的 GCC 版本。


**使用样例如下：**

.. code-block:: text
   
   import paddle
   from paddle.utils.cpp_extension import load

   custom_op_module = load(
       name="op_shared_libary_name",                # 生成动态链接库的名称
       sources=['relu_op.cc', 'relu_op.cu'],        # 自定义 OP 的源码文件列表
       extra_cxx_cflags=['-DPADDLE_WITH_MKLDNN'],   # 如预装的 Paddle 支持 MKLDNN，需指定此 flag
       extra_cuda_cflags=['-DPADDLE_WITH_MKLDNN'],  # 如预装的 Paddle 支持 MKLDNN，需指定此 flag
       interpreter='python3.7',                     # 可指定使用其他 python 解释器路径
       verbose=True                                 # 打印编译过程中的日志信息
   )

   x = paddle.randn([4, 10], dtype='float32')
   out = custom_op_module.relu(x)


参数：
  - **name** (str): 用于指定斌自定义 OP 编译后，生成的动态链接库的名字，不包括后缀如 .so 或者 .dll
  - **sources** (list[str]): 用于指定自定义 OP 对应的源码文件。cpp 源文件支持 .cc、.cpp 等后缀；cuda 源文件以 .cu 为后缀。
  - **extra_cxx_cflags** (list[str]): 用于指定编译 cpp 源文件时额外的编译选项。默认情况下，Paddle 框架相关的必要选项均已被隐式地包含；若预装的 Paddle 是支持 MKLDNN 的，则需要在此参数中额外指定 ``-DPADDLE_WITH_MKLDNN`` 。
  - **extra_cuda_cflags** (list[str]): 用于指定编译 cuda 源文件时额外的编译选项。默认情况下，Paddle 框架相关的必要选项均已被隐式地包含；若预装的 Paddle 是支持 MKLDNN 的，则需要在此参数中额外指定 ``-DPADDLE_WITH_MKLDNN`` 。 ``nvcc`` 相关的编译选项请参考：https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html
  - **extra_ldflags** (list[str]): 用于指定编译自定义 OP 时额外的链接选项。GCC 支持的链接选项请参考：https://gcc.gnu.org/onlinedocs/gcc/Link-Options.html
  - **extra_include_paths** (list[str]): 用于指定编译 cpp 或 cuda 源文件时，额外的头文件搜索目录。默认情况下，Paddle 框架相关头文件所在目录 ``site-packages/paddle/include`` 已被隐式地包含。
  - **build_directory** (str): 用于指定存放生成动态链接库的目录。若为 None， 则会使用环境变量 ``PADDLE_EXTENSION_DIR`` 的值作为默认的存放目录。可使用 ``paddle.utils.cpp_extension.get_build_directory()`` 接口查看当前的目录设置。
  - **interpreter** (str): 用于指定执行即时编译所需要的解释器的路径，支持别名和完整路径，默认值为 ``python`` 。若用户本地包含多个版本的 python 环境，需要确保默认的 ``python`` 命令与当前解释器一致，否则需要指定此参数。如当前为 python3.7 环境，可设置此参数为 ``'python3.7'`` 。
  - **verbose** (str): 用于指定是否需要输出编译过程中的日志信息

返回： 包含自定义 OP 的可调用 Module 对象。