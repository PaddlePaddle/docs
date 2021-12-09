.. _cn_api_paddle_tensor_lerp:

lerp
-------------------------------

.. py:function:: paddle.lerp(x, y, weight, name=None)
基于给定的 weight 计算 x 与 y 的线性插值

.. math::
    lerp(x, y, weight) = x + weight * (y - x)
参数
:::::::::

- **x**  (Tensor) - 输入的Tensor，作为线性插值开始的点，数据类型为：float32、float64。
- **y**  (Tensor) - 输入的Tensor，作为线性插值结束的点，数据类型为：float32、float64。
- **weight**  (float|Tensor) - 给定的权重值，weight为Tensor时数据类型为：float32、float64。
- **name**  (str，可选） - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name`。

返回
:::::::::

输出Tensor，与 ``x`` 数据类型相同。

代码示例
:::::::::

.. code-block:: python

    import paddle

    x = paddle.arange(1., 5., dtype='float32')
    y = paddle.empty([4], dtype='float32')
    y.fill_(10.)
    print(x)
    # Tensor(shape=[4], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
    #        [1., 2., 3., 4.])
    print(y)
    # Tensor(shape=[4], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
    #        [10., 10., 10., 10.])
    paddle.lerp(x, y, 0.5)
    # Tensor(shape=[4], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
    #        [5.5., 6., 6.5, 7.])
    paddle.lerp(x, y, paddle.full_like(x, 0.5))
    # Tensor(shape=[4], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
    #        [5.5., 6., 6.5, 7.])
