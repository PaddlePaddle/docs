.. _cn_api_paddle_framework_is_grad_enabled:

is_grad_enabled
-------------------------------

.. py:function:: paddle.is_grad_enabled()
获取当前动态图梯度计算模式。

返回
:::::::::

当前动态图梯度计算模式。

代码示例
:::::::::

.. code-block:: python
    
    import paddle

    paddle.is_grad_enabled() # True

    with paddle.set_grad_enabled(False):
        paddle.is_grad_enabled() # False

    paddle.enable_static()
    paddle.is_grad_enabled() # False