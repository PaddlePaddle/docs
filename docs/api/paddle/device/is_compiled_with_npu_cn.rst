.. _cn_api_fluid_is_compiled_with_npu:

is_compiled_with_npu
-------------------------------

.. py:function:: paddle.device.is_compiled_with_npu()

检查 ``whl`` 包是否可以被用来在 NPU 上运行模型。

返回：bool, 支持NPU则为True，否则为False。

**示例代码**

.. code-block:: python

    import paddle
    support_npu = paddle.device.is_compiled_with_npu()


