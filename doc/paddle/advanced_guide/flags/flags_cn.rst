
环境变量FLAGS
==================

调用说明
----------

PaddlePaddle中的环境变量FLAGS支持两种设置方式。

- 通过export来设置环境变量，如 :code:`export FLAGS_eager_delete_tensor_gb = 1.0` 。

- 通过API：:code:`get_flag` 和 :code:`set_flags` 来打印和设置环境变量FLAGS。API使用详情请参考 :ref:`cn_api_fluid_get_flags` 与 :ref:`cn_api_fluid_set_flags` 。


环境变量FLAGS功能分类
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
    others_cn.rst
