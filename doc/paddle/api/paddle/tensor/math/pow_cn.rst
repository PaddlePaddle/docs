.. _cn_api_paddle_tensor_math_pow:

pow
-------------------------------

.. py:function:: paddle.pow(x, y, name=None)



该OP是指数算子，逐元素计算 ``x`` 的 ``y`` 次幂。

.. math::

    out = x^{y}

参数
:::::::::
    - **x** （Tensor）- 多维 ``Tensor``，数据类型为 ``float32`` 、 ``float64`` 、 ``int32`` 或  ``int64`` 。
    - **y** （float|int|Tensor）- 如果类型是多维 ``Tensor``，其数据类型应该和 ``x`` 相同。
    - **name** （str, 可选）- 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name`。

返回
:::::::::
Tensor, 维度和数据类型都和 ``x`` 相同。


代码示例
:::::::::

.. code-block:: python

            import paddle

            x = paddle.to_tensor([1, 2, 3], dtype='float32')

            # example 1: y is a float or int
            res = paddle.pow(x, 2)
            print(res)
            # Tensor(shape=[3], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
            #        [1., 4., 9.])
            res = paddle.pow(x, 2.5)
            print(res)
            # Tensor(shape=[3], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
            #        [1.         , 5.65685415 , 15.58845711])

            # example 2: y is a Tensor
            y = paddle.to_tensor([2], dtype='float32')
            res = paddle.pow(x, y)
            print(res)
            # Tensor(shape=[3], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
            #        [1., 4., 9.])
