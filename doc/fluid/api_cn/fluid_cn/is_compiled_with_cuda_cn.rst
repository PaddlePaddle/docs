.. _cn_api_fluid_is_compiled_with_cuda:

is_compiled_with_cuda
-------------------------------

.. py:function:: paddle.fluid.is_compiled_with_cuda()

检查 ``whl`` 包是否可以被用来在GPU上运行模型

返回：支持gpu则为True,否则为False。

返回类型：out(boolean)

**示例代码**

.. code-block:: python

    import paddle.fluid as fluid
    support_gpu = fluid.is_compiled_with_cuda()


