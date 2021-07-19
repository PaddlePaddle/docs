.. _cn_api_paddle_framework_set_grad_enabled:

set_grad_enabled
-------------------------------

.. py:function:: paddle.set_grad_enabled(mode)


创建启用或禁用动态图梯度计算的上下文。


参数
:::::::::
    - mode (bool) - 启用或禁用动态图梯度计算。


返回
:::::::::
None


代码示例
:::::::::

..  code-block:: python

    import paddle

    x = paddle.ones([3, 2])
    x.stop_gradient = False
    with paddle.set_grad_enabled(False):
        y = x * 2
        with paddle.set_grad_enabled(True):
            z = x * 2
    print(y.stop_gradient)   # True
    print(z.stop_gradient)   # False

