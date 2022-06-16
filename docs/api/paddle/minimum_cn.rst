.. _cn_api_paddle_tensor_minimum:

minimum
-------------------------------

.. py:function:: paddle.minimum(x, y, name=None)


逐元素对比输入的两个 Tensor，并且把各个位置更小的元素保存到返回结果中。

等式是：

.. math::
        out = min(x, y)

.. note::
   ``paddle.minimum`` 遵守 broadcasting，如你想了解更多，请参见：ref:`cn_user_guide_broadcasting` 。

参数
:::::::::
   - **x** (Tensor) - 输入的 Tensor。数据类型为 ``float32`` 、 ``float64`` 、 ``int32`` 或  ``int64`` 。
   - **y** (Tensor) - 输入的 Tensor。数据类型为 ``float32`` 、 ``float64`` 、 ``int32`` 或  ``int64`` 。
   - **name** (str，可选) - 具体用法请参见：ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
   ``Tensor``。如果 x 和 y 有不同的 shape 且是可以广播的，返回 Tensor 的 shape 是 x 和 y 经过广播后的 shape。如果 x 和 y 有相同的 shape，返回 Tensor 的 shape 与 x，y 相同。


代码示例
::::::::::

.. code-block:: python

    import numpy as np
    import paddle

    x = paddle.to_tensor([[1, 2], [7, 8]])
    y = paddle.to_tensor([[3, 4], [5, 6]])
    res = paddle.minimum(x, y)
    print(res)
    #       [[1, 2],
    #        [5, 6]]

    x = paddle.to_tensor([[[1, 2, 3], [1, 2, 3]]])
    y = paddle.to_tensor([3, 0, 4])
    res = paddle.minimum(x, y)
    print(res)
    #       [[[1, 0, 3],
    #         [1, 0, 3]]]

    x = paddle.to_tensor([2, 3, 5], dtype='float32')
    y = paddle.to_tensor([1, np.nan, np.nan], dtype='float32')
    res = paddle.minimum(x, y)
    print(res)
    #       [ 1., nan, nan]

    x = paddle.to_tensor([5, 3, np.inf], dtype='float64')
    y = paddle.to_tensor([1, -np.inf, 5], dtype='float64')
    res = paddle.minimum(x, y)
    print(res)
    #       [   1., -inf.,    5.]
