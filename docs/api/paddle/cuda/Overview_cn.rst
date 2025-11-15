.. _cn_overview_cuda:

paddle.cuda
---------------------

paddle.cuda 目录下包含飞桨框架支持的 ``torch.cuda`` 兼容函数与模块接口

.. _about_cuda_funcs:

PyTorch 兼容函数
::::::::::::::::::::

.. csv-table::
    :header: "API 名称", "API 功能"
    :widths: 10, 30

    " :ref:`check_error <cn_api_paddle_cuda_check_error>` ", "检测 CUDA 错误码"
    " :ref:`cudart <cn_api_paddle_cuda_cudart>` ", "以模块的形式返回 CUDA Runtime 对象"
    " :ref:`current_stream <cn_api_paddle_cuda_current_stream>` ", "获取当前 CUDA 流"
    " :ref:`get_device_properties <cn_api_paddle_cuda_get_device_properties>` ", "获取 CUDA 设备属性"
    " :ref:`get_rng_state <cn_api_paddle_cuda_get_rng_state>` ", "获取随机数生成器状态"
    " :ref:`is_available <cn_api_paddle_cuda_is_available>` ", "检查 CUDA 是否可用"
    " :ref:`is_initialized <cn_api_paddle_cuda_is_initialized>` ", "判断 CUDA 是否已经初始化"
    " :ref:`manual_seed_all <cn_api_paddle_cuda_manual_seed_all>` ", "设置全局随机种子"
    " :ref:`mem_get_info <cn_api_paddle_cuda_mem_get_info>` ", "获取指定设备上的全局空闲显存和显存总量"
    " :ref:`set_rng_state <cn_api_paddle_cuda_set_rng_state>` ", "设置随机数生成器状态"
    " :ref:`synchronize <cn_api_paddle_cuda_synchronize>` ", "同步 CUDA 设备"
    " :ref:`current_device <cn_api_paddle_cuda_current_device>` ", "返回当前设备的索引"
    " :ref:`device_count <cn_api_paddle_cuda_device_count>` ", "返回可用的 CUDA 设备数量"
    " :ref:`empty_cache <cn_api_paddle_cuda_empty_cache>` ", "释放当前设备上所有未占用的缓存内存"
    " :ref:`memory_allocated <cn_api_paddle_cuda_memory_allocated>` ", "返回当前设备上分配的内存总量"
    " :ref:`memory_reserved <cn_api_paddle_cuda_memory_reserved>` ", "返回当前设备上由缓存分配器管理的内存总量"
    " :ref:`set_device <cn_api_paddle_cuda_set_device>` ", "设置当前设备"
    " :ref:`Stream <cn_api_paddle_cuda_Stream>` ", "CUDA 流类"
    " :ref:`get_stream_from_external <cn_api_paddle_cuda_get_stream_from_external>` ", "从外部流创建 Paddle 流"
    " :ref:`device <cn_api_paddle_cuda_device>` ", "临时选择设备使用"
    " :ref:`manual_seed <cn_api_paddle_cuda_manual_seed>` ", "设置设备随机种子"
    " :ref:`max_memory_allocated <cn_api_paddle_cuda_max_memory_allocated>` ", "获取最大内存分配量"
    " :ref:`reset_peak_memory_stats <cn_api_paddle_cuda_reset_peak_memory_stats>` ", "重置峰值内存统计"
    " :ref:`get_device_capability <cn_api_paddle_cuda_get_device_capability>` ", "返回指定设备的计算能力"
