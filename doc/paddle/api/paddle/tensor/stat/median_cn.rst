.. _cn_api_tensor_cn_median:

median
-------------------------------

.. py:function:: paddle.median(x, axis=None, keepdim=False, name=None)

沿给定的轴 ``axis`` 计算 ``x`` 中元素的中位数。

参数
::::::::::
   - x (Tensor) - 输入的Tensor，数据类型为：bool、float16、float32、float64、int32、int64。
   - axis (int, 可选) - 指定对 ``x`` 进行计算的轴。``axis`` 可以是int。``axis`` 值应该在范围[-D, D)内，D是 ``x`` 的维度。如果 ``axis`` 或者其中的元素值小于0，则等价于 :math:`axis + D` 。如果 ``axis`` 是None，则对 ``x`` 的全部元素计算中位数。默认值为None。
   - keepdim (bool, 可选) - 是否在输出Tensor中保留减小的维度。如果 ``keepdim`` 为True，则输出Tensor和 ``x`` 具有相同的维度(减少的维度除外，减少的维度的大小为1)。否则，输出Tensor的形状会在 ``axis`` 上进行squeeze操作。默认值为False。
   - name (str, 可选) - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name`。

返回
::::::::::
    ``Tensor`` ，沿着 ``axis`` 进行中位数计算的结果。如果``x`` 的数据类型为float64，则返回值的数据类型为float64，反之返回值数据类型为float32。

代码示例
::::::::::

.. code-block:: python

    import paddle

    x = paddle.arange(12).reshape([3, 4])
    # x is [[0 , 1 , 2 , 3 ],
    #       [4 , 5 , 6 , 7 ],
    #       [8 , 9 , 10, 11]]

    y1 = paddle.median(x)
    # y1 is [5.5]

    y2 = paddle.median(x, axis=0)
    # y2 is [4., 5., 6., 7.]

    y3 = paddle.median(x, axis=1)
    # y3 is [1.5, 5.5, 9.5]

    y4 = paddle.median(x, axis=0, keepdim=True)
    # y4 is [[4., 5., 6., 7.]]
