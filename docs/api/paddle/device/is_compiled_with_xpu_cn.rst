.. _cn_api_fluid_is_compiled_with_xpu:

is_compiled_with_xpu
-------------------------------

.. py:function:: paddle.device.is_compiled_with_xpu()




检查 ``whl`` 包是否可以被用来在Baidu Kunlun XPU上运行模型

返回：bool，支持Baidu Kunlun XPU则为True，否则为False。

**示例代码**

.. code-block:: python

    import paddle
    support_xpu = paddle.paddle.is_compiled_with_xpu()


