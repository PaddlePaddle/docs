.. _cn_api_paddle_utils_cpp_extension_setup:

setup
-------------------------------

.. py:function:: paddle.utils.cpp_extension.setup(**attr)

此接口用于配置如何编译自定义 OP 源文件，包括编译动态库，自动地生成 Python API 并以 Module 的形式安装到 site-packages 目录等过程。编译完成后，支持通过 ``import`` 语句导入使用。

此接口是对 Python 内建库中的 ``setuptools.setup`` 接口的进一步封装，支持的参数类型，以及使用方式均与原生接口保持一致。接口隐藏了 Paddle 框架内部概念，如默认需要指定的编译选项，头文件搜索目录，链接选项等；此接口会自动搜索和检查本地的 ``cc`` 和 ``nvcc`` 编译命令和版本环境，根据用户指定的 ``Extension`` 类型，完成支持 CPU 或 GPU 设备的算子编译。

同时，编译前会执行 `ABI 兼容性检查 <https://gcc.gnu.org/onlinedocs/libstdc++/manual/abi.html>`_ ，即检查编译器版本是否与本地安装的 Paddle 一致。如在 Linux 下，对于 CUDA 10.1 以上的 Paddle 默认使用 GCC 8.2 编译，则本地 ``cc`` 对应的编译器版本也需为 8.2，在 Windows 下，Paddle 使用 Visualt Studio 2017 编译，则本地也需安装
大于 2017 的 Visual Studio，如果不满足，则可能由于 ABI 兼容性原因引发自定义 OP 编译或执行报错。Mac 下默认使用 clang 进行编译，无 ABI 兼容性问题。

相对于即时编译的 :ref:`cn_api_paddle_utils_cpp_extension_load` 接口，此接口仅需执行一次 ``python setup.py install`` 命令，即可像其他 python 库一样 import 导入使用。如下是一个 ``setup.py`` 文件的简单样例：


.. note::

    1. 目前已支持 Linux ，MacOS 和 Windows 平台。
    2. 编译器的 ABI 兼容性是向前兼容的，Linux 下推荐使用 GCC 8.2 高版本作为 ``/usr/bin/cc`` 命令的软链对象，可通过 ``which cc`` 查看 ``cc`` 命令的位置，使用 ``cc --version`` 查看对应的 GCC 版本。
    3. Windows 下需要安装 Visual Studio 2017 及以上版本。

.. code-block:: text

    # setup.py

    # 方式一：编译支持 CPU 和 GPU 的算子
    from paddle.utils.cpp_extension import CUDAExtension, setup

    setup(
        name='custom_op',  # package 的名称，用于 import
        ext_modules=CUDAExtension(
            sources=['relu_op.cc', 'relu_op.cu', 'tanh_op.cc', 'tanh_op.cu']  # 支持同时编译多个 OP
        )
    )

    # 方式二：编译支持仅 CPU 的算子
    from paddle.utils.cpp_extension import CppExtension, setup

    setup(
        name='custom_op',  # package 的名称，用于 import
        ext_modules=CppExtension(
            sources=['relu_op.cc', 'tanh_op.cc']  # 支持同时编译多个 OP
        )
    )



在源文件所在目录下执行 ``python setup.py install`` 即可完成自定义 OP 编译和 ``custom_op`` 库的安装。在组网时，可以通过如下方式使用：

.. code-block:: text

    import paddle
    from custom_op import relu, tanh

    x = paddle.randn([4, 10], dtype='float32')
    relu_out = relu(x)
    tanh_out = tanh(x)



参数
::::::::::::

  - **name** (string) - 用于指定生成的动态链接库的名称，以及安装到 site-packages 的 ``Module`` 名字
  - **ext_modules** (Extension) - 用于指定包含自定义 OP 必要源文件、编译选项等信息的 ``Extension``。若只编译运行在 CPU 设备上的 OP，请使用 :ref:`cn_api_paddle_utils_cpp_extension_CppExtension`；若编译同时支持 GPU 设备上的 OP，请使用 :ref:`cn_api_paddle_utils_cpp_extension_CUDAExtension` 。
  - **include_dirs** (list[str]，可选) - 用于指定编译自定义 OP 时额外的头文件搜索目录。此接口默认会自动添加 ``site-packages/paddle/include`` 目录。若自定义 OP 源码引用了其他三方库文件，可以通过此参数指定三方库的搜索目录。默认值为 None 。
  - **extra_compile_args** (list[str] | dict，可选) - 用于指定编译自定义 OP 时额外的编译选项，如 ``-O3`` 等。若为 ``list[str]`` 类型，则表示这些编译选项会同时应用到 ``cc`` 和 ``nvcc`` 编译过程；可以通过 ``{'cxx': [...], 'nvcc': [...]}`` 字典的形式单独指定额外的 ``cc`` 或 ``nvcc`` 的编译选项。默认值为 None 。
  - **\*\*attr** (dict，可选) - 其他参数与 ``setuptools.setup`` 一致。

返回
::::::::::::

无。
