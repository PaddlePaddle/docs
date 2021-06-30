.. _cn_api_fluid_is_compiled_with_cuda:

is_compiled_with_cuda
-------------------------------

.. py:function:: paddle.is_compiled_with_cuda()




检查 ``whl`` 包是否可以被用来在GPU上运行模型

返回：支持GPU则为True，否则为False。

返回：是否支持GPU的bool值

**示例代码**

.. code-block:: python

    import paddle
    support_gpu = paddle.is_compiled_with_cuda()


