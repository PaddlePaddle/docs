.. _cn_overview_utils:

paddle.utils
---------------------

paddle.utils 目录下包含飞桨框架工具类的 API。具体如下：

-  :ref:`自定义 OP 相关 API <about_cpp_extension>`
-  :ref:`工具类相关 API <about_utils>`



.. _about_cpp_extension:

自定义 OP 相关 API
::::::::::::::::::::

.. csv-table::
    :header: "API 名称", "API 功能"
    :widths: 10, 30

    " :ref:`load <cn_api_paddle_utils_cpp_extension_load>` ", "飞桨框架一键编译自定义 OP、自动生成和返回 Python API 的接口"
    " :ref:`setup <cn_api_paddle_utils_cpp_extension_setup>` ", "飞桨框架编译自定义 OP、并安装到 site-package 目录的接口"
    " :ref:`CppExtension <cn_api_paddle_utils_cpp_extension_CppExtension>` ", "飞桨框架编译仅支持 CPU 的自定义 OP 扩展类"
    " :ref:`CUDAExtension <cn_api_paddle_utils_cpp_extension_CUDAExtension>` ", "飞桨框架编译支持 GPU 的自定义 OP 扩展类"
    " :ref:`get_build_directory <cn_api_paddle_utils_cpp_extension_get_build_directory>` ", "返回一键编译自定义 OP 的 build 目录"


.. _about_utils:

工具类相关 API
::::::::::::::::::::

.. csv-table::
    :header: "API 名称", "API 功能"
    :widths: 11, 30

    " :ref:`deprecated <cn_api_paddle_utils_deprecated>` ", "飞桨框架废弃 API 装饰器"
    " :ref:`get_weights_path_from_url <cn_api_paddle_utils_download_get_weights_path_from_url>` ", "从文件夹获取权重"
    " :ref:`run_check <cn_api_paddle_utils_run_check>` ", "检查是否正常安装飞桨框架"
    " :ref:`generate <cn_api_paddle_utils_unique_name_generate>` ", "产生以前缀开头的唯一名称"
    " :ref:`guard <cn_api_paddle_utils_unique_name_guard>` ", "更改命名空间"
    " :ref:`switch <cn_api_paddle_utils_unique_name_switch>` ", "切换命名空间"
    " :ref:`Profiler <cn_api_paddle_profiler_Profiler>` ", "通用性能分析器"
    " :ref:`require_version <cn_api_paddle_utils_require_version>` ", "用于检查已安装的飞桨版本是否介于[min_version, max_version]之间"
    " :ref:`to_dlpack <cn_api_paddle_utils_dlpack_to_dlpack>` ", "用于将 Tensor 对象转换为 DLPack"
    " :ref:`from_dlpack <cn_api_paddle_utils_dlpack_from_dlpack>` ", "用于从 DLPack 中解码出 Tensor 对象"
    " :ref:`try_import <cn_api_paddle_utils_try_import>` ", "用于尝试导入一个模块并在失败时提供自定义错误信息"