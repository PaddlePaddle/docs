.. _cn_overview_utils:

paddle.utils
---------------------

paddle.utils 目录下包含飞桨框架工具类的API。具体如下：

-  :ref:`自定义OP相关API <about_cpp_extension>`
-  :ref:`工具类相关API <about_utils>`



.. _about_cpp_extension:

自定义OP相关API
::::::::::::::::::::

.. csv-table::
    :header: "API名称", "API功能"
    :widths: 10, 30

    " :ref:`load <cn_api_paddle_utils_cpp_extension_load>` ", "飞桨框架一键编译自定义OP、自动生成和返回Python API的接口"
    " :ref:`setup <cn_api_paddle_utils_cpp_extension_setup>` ", "飞桨框架编译自定义OP、并安装到site-package目录的接口"
    " :ref:`CppExtension <cn_api_paddle_utils_cpp_extension_CppExtension>` ", "飞桨框架编译仅支持CPU的自定义OP扩展类"
    " :ref:`CUDAExtension <cn_api_paddle_utils_cpp_extension_CUDAExtension>` ", "飞桨框架编译支持GPU的自定义OP扩展类"
    " :ref:`get_build_directory <cn_api_paddle_utils_cpp_extension_get_build_directory>` ", "返回一键编译自定义OP的build目录"


.. _about_utils:

工具类相关API
::::::::::::::::::::

.. csv-table::
    :header: "API名称", "API功能"
    :widths: 10, 30

    " :ref:`deprecated <cn_api_paddle_utils_deprecated>` ", "飞桨框架废弃API装饰器"
    " :ref:`get_weights_path_from_url <cn_api_paddle_utils_download_get_weights_path_from_url>` ", "从文件夹获取权重"
    " :ref:`run_check <cn_api_paddle_utils_run_check>` ", "检查是否正常安装飞桨框架"
    " :ref:`generate <cn_api_fluid_unique_name_generate>` ", "产生以前缀开头的唯一名称"
    " :ref:`guard <cn_api_fluid_unique_name_guard>` ", "更改命名空间"
    " :ref:`switch <cn_api_fluid_unique_name_switch>` ", "切换命名空间"
    " :ref:`cuda_profiler <cn_api_fluid_profiler_cuda_profiler>` ", "CUDA性能分析器"
    " :ref:`profiler <cn_api_fluid_profiler_profiler>` ", "通用性能分析器"
    " :ref:`reset_profiler <cn_api_fluid_profiler_reset_profiler>` ", "清除之前的性能分析记录"
    " :ref:`start_profiler <cn_api_fluid_profiler_start_profiler>` ", "激活使用性能分析器"
    " :ref:`stop_profiler <cn_api_fluid_profiler_stop_profiler>` ", "停止使用性能分析器"
    " :ref:`require_version <cn_api_fluid_require_version>` ", "用于检查已安装的飞桨版本是否介于[min_version, max_version]之间"
    " :ref:`to_dlpack <cn_api_paddle_utils_dlpack_to_dlpack>` ", "用于将Tensor对象转换为DLPack"
    " :ref:`from_dlpack <cn_api_paddle_utils_dlpack_from_dlpack>` ", "用于从DLPack中解码出Tensor对象"
