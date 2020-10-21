.. _cn_api_paddle_tensor_minimum:

minimum
-------------------------------

.. py:function:: paddle.tensor.minimum(x, y, axis=-1, name=None)


该OP逐元素对比输入的两个多维Tensor，并且把各个位置更小的元素保存到返回结果中。

等式是：

.. math::
        Out = min(X, Y)

- :math:`X` ：多维Tensor。
- :math:`Y` ：多维Tensor。

此运算算子有两种情况：
        1. :math:`Y` 的 ``shape`` 与 :math:`X` 相同。
        2. :math:`Y` 的 ``shape`` 是 :math:`X` 的连续子序列。

对于情况2：
        1. 用 :math:`Y` 的 ``shape`` 匹配 :math:`X` 的 ``shape``，其中 ``axis`` 是 :math:`Y` 在 :math:`X` 上的起始维度的位置。
        2. 如果 ``axis`` < 0（默认值为-1），则 :math:`axis = abs(X.ndim - Y.ndim) - axis - 1` 。
        3. 考虑到子序列， :math:`Y` 的大小为1的尾部维度将被忽略，例如shape（Y）=（2,1）=>（2）。

例如：

..  code-block:: text

        shape(X) = (2, 3, 4, 5), shape(Y) = (,)
        shape(X) = (2, 3, 4, 5), shape(Y) = (5,)
        shape(X) = (2, 3, 4, 5), shape(Y) = (4, 5), with axis=-1(default) or axis=2
        shape(X) = (2, 3, 4, 5), shape(Y) = (3, 4), with axis=1
        shape(X) = (2, 3, 4, 5), shape(Y) = (2), with axis=0
        shape(X) = (2, 3, 4, 5), shape(Y) = (2, 1), with axis=0

具体的飞桨的广播（broadcasting）机制可以参考 `<<PaddlePaddle广播机制文档>> <https://github.com/PaddlePaddle/FluidDoc/blob/develop/doc/fluid/beginners_guide/basic_concept/broadcasting.rst>`_ 。

参数
:::::::::
   - **x** （Tensor）- 多维Tensor。数据类型为 ``float32`` 、 ``float64`` 、 ``int32`` 或  ``int64`` 。
   - **y** （Tensor）- 多维Tensor。数据类型为 ``float32`` 、 ``float64`` 、 ``int32`` 或  ``int64`` 。
   - **axis** （int32, 可选）- Y的维度对应到X维度上时的索引。默认值为 -1。
   - **name** （string, 可选）- 输出的名字。默认值为None。该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` 。

返回
:::::::::
   Tensor，维度和数据类型与 ``x`` 相同的多维Tensor。


代码示例
::::::::::

.. code-block:: python

    import paddle
    paddle.disable_static()
  
    x = paddle.to_tensor([[1, 2], [3, 4]], dtype='float32')
    y = paddle.to_tensor([[5, 6], [7, 8]], dtype='float32')
    res = paddle.minimum(x, y)
    print(res.numpy())
    #[[1. 2.]
    # [3. 4.]]

    x = paddle.to_tensor([[[1, 2, 3], [1, 2, 3]]], dtype='float32')
    y = paddle.to_tensor([1, 2], dtype='float32')
    res = paddle.minimum(x, y, axis=1)
    print(res.numpy())
    #[[[1. 1. 1.]
    #  [2. 2. 2.]]]

    x = paddle.to_tensor([2, 3, 5], dtype='float32')
    y = paddle.to_tensor([1, 4, np.nan], dtype='float32')
    res = paddle.minimum(x, y)
    print(res.numpy())
    #[ 1.  3. nan]

    x = paddle.to_tensor([5, 3, np.inf], dtype='float32')
    y = paddle.to_tensor([1, 4, 5], dtype='float32')
    res = paddle.minimum(x, y)
    print(res.numpy())
    #[1. 3. 5.]
