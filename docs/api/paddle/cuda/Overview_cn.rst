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
    " :ref:`is_initialized <cn_api_paddle_cuda_is_initialized>` ", "判断 CUDA 是否已经初始化"
    " :ref:`mem_get_info <cn_api_paddle_cuda_mem_get_info>` ", "获取指定设备上的全局空闲显存和显存总量"
    " :ref:`current_device <cn_api_paddle_cuda_current_device>` ", "返回当前设备的索引"
    " :ref:`device_count <cn_api_paddle_cuda_device_count>` ", "返回可用的 CUDA 设备数量"
    " :ref:`empty_cache <cn_api_paddle_cuda_empty_cache>` ", "释放当前设备上所有未占用的缓存内存"
    " :ref:`memory_allocated <cn_api_paddle_cuda_memory_allocated>` ", "返回当前设备上分配的内存总量"
    " :ref:`memory_reserved <cn_api_paddle_cuda_memory_reserved>` ", "返回当前设备上由缓存分配器管理的内存总量"
    " :ref:`set_device <cn_api_paddle_cuda_set_device>` ", "设置当前设备"
