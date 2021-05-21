.. _cn_api_fluid_is_compiled_with_rocm:

is_compiled_with_rocm
-------------------------------

.. py:function:: paddle.is_compiled_with_rocm()




检查 ``whl`` 包是否可以被用来在AMD或海光GPU(ROCm)上运行模型

返回：支持GPU(ROCm)则为True，否则为False。

返回：是否支持GPU(ROCm)的bool值

**示例代码**

.. code-block:: python

    import paddle
    support_gpu = paddle.is_compiled_with_rocm()


