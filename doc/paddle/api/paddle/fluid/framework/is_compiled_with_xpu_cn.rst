.. _cn_api_fluid_is_compiled_with_xpu:

is_compiled_with_xpu
-------------------------------

.. py:function:: paddle.is_compiled_with_xpu()




检查 ``whl`` 包是否可以被用来在Baidu Kunlun XPU上运行模型

返回：支持Baidu Kunlun XPU则为True，否则为False。

返回：是否支持Baidu Kunlun XPU的bool值

**示例代码**

.. code-block:: python

    import paddle
    support_xpu = paddle.is_compiled_with_xpu()


