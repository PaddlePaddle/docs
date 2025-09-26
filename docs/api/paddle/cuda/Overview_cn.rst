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
