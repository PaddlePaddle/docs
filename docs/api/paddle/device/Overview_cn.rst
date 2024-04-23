.. _cn_overview_device:

paddle.device
---------------------

`paddle.device` 模块提供了一系列与设备相关的 API，用于管理和配置计算设备。具体如下：

-  :ref:`设备设置与属性获取 <cn_device_setting>`
-  :ref:`编译环境检测 <cn_device_compile>`
-  :ref:`设备描述符 <cn_device_descriptor>`
-  :ref:`Stream 与 Event 辅助类 <cn_device_stream_event>`
-  :ref:`Stream 与 Event 相关 API <cn_device_stream_event_api>`

paddle.device 目录下包含 cuda 目录， cuda 目录中存放 CUDA 相关的 API。具体如下：

-  :ref:`CUDA 相关 <cn_device_cuda>`

.. _cn_device_setting:

设备设置与属性获取
::::::::::::::::::::

.. csv-table::
    :header: "API 名称", "API 功能"
    :widths: 10, 30

    " :ref:`get_all_custom_device_type <cn_api_paddle_device_get_all_custom_device_type>` ", "获得所有可用的自定义设备类型"
    " :ref:`get_all_device_type <cn_api_paddle_device_get_all_device_type>` ", "获得所有可用的设备类型"
    " :ref:`get_available_custom_device <cn_api_paddle_device_get_available_custom_device>` ", "获得所有可用的自定义设备"
    " :ref:`get_available_device <cn_api_paddle_device_get_available_device>` ", "获得所有可用的设备"
    " :ref:`get_cudnn_version <cn_api_paddle_device_get_cudnn_version>` ", "获得 cudnn 的版本"
    " :ref:`set_device <cn_api_paddle_device_set_device>` ", "指定 OP 运行的全局设备"
    " :ref:`get_device <cn_api_paddle_device_get_device>` ", "获得 OP 运行的全局设备"

.. _cn_device_compile:

编译环境检测
::::::::::::::::::::

.. csv-table::
    :header: "API 名称", "API 功能"
    :widths: 10, 30

    " :ref:`is_compiled_with_cinn <cn_api_paddle_device_is_compiled_with_cinn>` ", "检查 ``whl`` 包是否可以被用来在 CINN 上运行模型"
    " :ref:`is_compiled_with_cuda <cn_api_paddle_device_is_compiled_with_cuda>` ", "检查 ``whl`` 包是否可以被用来在 GPU 上运行模型"
    " :ref:`is_compiled_with_custom_device <cn_api_paddle_device_is_compiled_with_custom_device>` ", "检查 ``whl`` 包是否可以被用来在指定类型的自定义新硬件上运行模型"
    " :ref:`is_compiled_with_ipu <cn_api_paddle_device_is_compiled_with_ipu>` ", "检查 ``whl`` 包是否可以被用来在 Graphcore IPU 上运行模型"
    " :ref:`is_compiled_with_mlu <cn_api_paddle_device_is_compiled_with_mlu>` ", "检查 ``whl`` 包是否可以被用来在 Cambricon MLU 上运行模型"
    " :ref:`is_compiled_with_npu <cn_api_paddle_device_is_compiled_with_npu>` ", "检查 ``whl`` 包是否可以被用来在 NPU 上运行模型"
    " :ref:`is_compiled_with_rocm <cn_api_paddle_device_is_compiled_with_rocm>` ", "检查 ``whl`` 包是否可以被用来在 AMD 或海光 GPU(ROCm) 上运行模型"
    " :ref:`is_compiled_with_xpu <cn_api_paddle_device_is_compiled_with_xpu>` ", "检查 ``whl`` 包是否可以被用来在 Baidu Kunlun XPU 上运行模型"

.. _cn_device_descriptor:

设备描述符
::::::::::::::::::::

.. csv-table::
    :header: "API 名称", "API 功能"
    :widths: 10, 30

    " :ref:`IPUPlace <cn_api_paddle_device_IPUPlace>` ", "``IPUPlace`` 是一个设备描述符，指定 ``IPUPlace`` 则模型将会运行在该设备上"
    " :ref:`MLUPlace <cn_api_paddle_device_MLUPlace>` ", "``MLUPlace`` 是一个设备描述符，指定 ``MLUPlace`` 则模型将会运行在该设备上"
    " :ref:`XPUPlace <cn_api_paddle_device_XPUPlace>` ", "``XPUPlace`` 是一个设备描述符，表示一个分配或将要分配 ``Tensor`` 的 Baidu Kunlun XPU 设备"

.. _cn_device_stream_event:

Stream 与 Event 辅助类
::::::::::::::::::::

.. csv-table::
    :header: "类名称", "辅助类功能"
    :widths: 10, 30

    " :ref:`Stream <cn_api_paddle_device_Stream>` ", "``StreamBase`` 的设备流包装器"
    " :ref:`Event <cn_api_paddle_device_Event>` ", "``StreamBase`` 的设备事件包装器"


.. _cn_device_stream_event_api:

Stream 与 Event 相关 API
::::::::::::::::::::

.. csv-table::
    :header: "API 名称", "API 功能"
    :widths: 10, 30

    " :ref:`current_stream <cn_api_paddle_device_current_stream>` ", "通过 device 返回当前的 stream"
    " :ref:`set_stream <cn_api_paddle_device_set_stream>` ", "设置当前的 stream"
    " :ref:`stream_guard <cn_api_paddle_device_stream_guard>` ", "切换当前的 stream 为输入指定的 stream，该 API 目前仅支持动态图模式"
    " :ref:`synchronize <cn_api_paddle_device_synchronize>` ", "等待给定的设备上的计算完成"


.. _cn_device_cuda:

CUDA 相关
::::::::::::::::::::

.. csv-table::
    :header: "API 名称", "API 功能"
    :widths: 10, 30

    " :ref:`Stream <cn_api_paddle_device_cuda_Stream>` ", "CUDA ``StreamBase`` 的设备流包装器，该 API 未来计划废弃，不推荐使用"
    " :ref:`Event <cn_api_paddle_device_cuda_Event>` ", "CUDA ``StreamBase`` 的设备事件包装器，该 API 未来计划废弃，不推荐使用"
    " :ref:`current_stream <cn_api_paddle_device_cuda_current_stream>` ", "通过 device 返回当前的 CUDA stream"
    " :ref:`device_count <cn_api_paddle_device_cuda_device_count>` ", "返回值是 int，表示当前程序可用的 GPU 数量"
    " :ref:`empty_cache <cn_api_paddle_device_cuda_empty_cache>` ", "用于释放显存分配器中空闲的显存"
    " :ref:`get_device_capability <cn_api_paddle_device_cuda_get_device_capability>` ", "获取 CUDA 设备计算能力的主要和次要修订号"
    " :ref:`get_device_name <cn_api_paddle_device_cuda_get_device_name>` ", "获取 CUDA 设备名称"
    " :ref:`get_device_properties <cn_api_paddle_device_cuda_get_device_properties>` ", "获取 CUDA 设备属性"
    " :ref:`max_memory_allocated <cn_api_paddle_device_cuda_max_memory_allocated>` ", "返回给定设备上分配给 Tensor 的显存峰值"
    " :ref:`max_memory_reserved <cn_api_paddle_device_cuda_max_memory_reserved>` ", "返回给定设备上由 Allocator 管理的显存峰值"
    " :ref:`memory_allocated <cn_api_paddle_device_cuda_memory_allocated>` ", "返回给定设备上当前分配给 Tensor 的显存大小"
    " :ref:`memory_reserved <cn_api_paddle_device_cuda_memory_reserved>` ", "返回给定设备上当前由 Allocator 管理的显存大小"
    " :ref:`stream_guard <cn_api_paddle_device_cuda_stream_guard>` ", "切换当前的 CUDA stream 为输入指定的 stream，该 API 目前仅支持动态图模式"
    " :ref:`synchronize <cn_api_paddle_device_cuda_synchronize>` ", "等待给定的 CUDA 设备上的计算完成"
