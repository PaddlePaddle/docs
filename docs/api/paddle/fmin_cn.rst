.. _cn_api_paddle_tensor_fmin:

fmin
-------------------------------

.. py:function:: paddle.fmin(x, y, name=None)


比较两个Tensor对应位置的元素，返回一个包含该元素最小值的新Tensor。如果两个元素其中一个是nan值，则直接返回另一个值，如果两者都是nan值，则返回第一个nan值。

等式是：

.. math::
        out = fmin(x, y)

.. note::
   ``paddle.fmin`` 遵守broadcasting，如您想了解更多，请参见 :ref:`cn_user_guide_broadcasting` 。

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
    y = paddle.to_tensor([[3, 4], [5, 6]])
    res = paddle.fmin(x, y)
    print(res)
    #       [[1, 2],
    #        [5, 6]]

    x = paddle.to_tensor([[[1, 2, 3], [1, 2, 3]]])
    y = paddle.to_tensor([3, 0, 4])
    res = paddle.fmin(x, y)
    print(res)
    #       [[[1, 0, 3],
    #         [1, 0, 3]]]

    x = paddle.to_tensor([2, 3, 5], dtype='float32')
    y = paddle.to_tensor([1, np.nan, np.nan], dtype='float32')
    res = paddle.fmin(x, y)
    print(res)
    #       [ 1., 3., 5.]

    x = paddle.to_tensor([5, 3, np.inf], dtype='float64')
    y = paddle.to_tensor([1, -np.inf, 5], dtype='float64')
    res = paddle.fmin(x, y)
    print(res)
    #       [   1., -inf.,    5.]
