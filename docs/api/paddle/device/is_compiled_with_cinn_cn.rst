.. _cn_api_fluid_is_compiled_with_cinn:

is_compiled_with_cinn
-------------------------------

.. py:function:: paddle.device.is_compiled_with_cinn()

检查 ``whl`` 包是否可以被用来在 CINN 上运行模型。

返回：bool，支持CINN则为True，否则为False。

**示例代码**

.. code-block:: python

    import paddle
    support_cinn = paddle.device.is_compiled_with_cinn()


