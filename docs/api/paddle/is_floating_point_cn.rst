.. _cn_api_paddle_tensor_is_floating_point:

is_floating_point
-------------------------------

.. py:function:: paddle.is_floating_point(x)
判断输入Tensor是否为浮点类型。

参数
:::::::::

- **x**  (Tensor) - 输入的Tensor。

返回
:::::::::

输入Tensor是否为浮点类型。

代码示例
:::::::::

.. code-block:: python
    
    import paddle

    x = paddle.arange(1., 5., dtype='float32')
    y = paddle.arange(1, 5, dtype='int32')
    print(paddle.is_floating_point(x))
    # True
    print(paddle.is_floating_point(y))
    # False