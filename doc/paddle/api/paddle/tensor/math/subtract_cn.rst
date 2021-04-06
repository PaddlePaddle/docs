.. _cn_api_paddle_tensor_subtract:

subtract
-------------------------------

.. py:function:: paddle.subtract(x, y, name=None)


该OP是逐元素相减算子，输入 ``x`` 与输入 ``y`` 逐元素相减，并将各个位置的输出元素保存到返回结果中。

等式是：

.. math::
        out = x - y

.. note::
   ``paddle.subtract`` 遵守broadcasting，如您想了解更多，请参见 :ref:`cn_user_guide_broadcasting` 。

参数
:::::::::
   - **x** （Tensor）- 输入的Tensor。数据类型为 ``float32`` 、 ``float64`` 、 ``int32`` 或  ``int64`` 。
   - **y** （Tensor）- 输入的Tensor。数据类型为 ``float32`` 、 ``float64`` 、 ``int32`` 或  ``int64`` 。
   - **name** （str, 可选）- 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name`。

返回
:::::::::
   ``Tensor``，存储运算后的结果。如果x和y有不同的shape且是可以广播的，返回Tensor的shape是x和y经过广播后的shape。如果x和y有相同的shape，返回Tensor的shape与x，y相同。


代码示例
::::::::::

.. code-block:: python

    import numpy as np
    import paddle

    x = paddle.to_tensor([[1, 2], [7, 8]])
    y = paddle.to_tensor([[5, 6], [3, 4]])
    res = paddle.subtract(x, y)
    print(res)
    #       [[-4, -4],
    #        [4, 4]]

    x = paddle.to_tensor([[[1, 2, 3], [1, 2, 3]]])
    y = paddle.to_tensor([1, 0, 4])
    res = paddle.subtract(x, y)
    print(res)
    #       [[[ 0,  2, -1],
    #         [ 0,  2, -1]]]

    x = paddle.to_tensor([2, np.nan, 5], dtype='float32')
    y = paddle.to_tensor([1, 4, np.nan], dtype='float32')
    res = paddle.subtract(x, y)
    print(res)
    #       [ 1., nan, nan]

    x = paddle.to_tensor([5, np.inf, -np.inf], dtype='float64')
    y = paddle.to_tensor([1, 4, 5], dtype='float64')
    res = paddle.subtract(x, y)
    print(res)
    #       [   4.,  inf., -inf.]
