.. _cn_api_fluid_is_compiled_with_npu:

is_compiled_with_npu
-------------------------------

.. py:function:: paddle.fluid.is_compiled_with_npu()
检查 ``whl`` 包是否可以被用来在 NPU 上运行模型。

返回：支持 NPU 则为True，否则为False。

返回：是否支持 NPU 的bool值

**示例代码**

.. code-block:: python
    import paddle
    support_npu  = paddle.is_compiled_with_npu()

