
.. _cn_guides_flags_flags:

环境变量 FLAGS
==================

调用说明
----------

PaddlePaddle 中的环境变量 FLAGS 支持两种设置方式。

- 通过 export 来设置环境变量，如 :code:`export FLAGS_eager_delete_tensor_gb = 1.0` 。

- 通过 API：:code:`get_flag` 和 :code:`set_flags` 来打印和设置环境变量 FLAGS。API 使用详情请参考 :ref:`cn_api_paddle_get_flags` 与 :ref:`cn_api_paddle_set_flags` 。


环境变量 FLAGS 功能分类
----------------------

..  toctree::
    :maxdepth: 1

    cudnn_cn.rst
    data_cn.rst
    debug_cn.rst
    device_cn.rst
    distributed_cn.rst
    executor_cn.rst
    memory_cn.rst
    npu_cn.rst
    others_cn.rst
