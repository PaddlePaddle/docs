.. _cn_api_paddle_nanmedian:

nanmedian
-------------------------------

.. py:function:: paddle.nanmedian(x, axis=None, keepdim=False, mode='avg', name=None)

沿给定的轴 ``axis`` 计算中位数，同时忽略 NAN 元素。
如果元素的有效计数为偶数，则计算并返回中间两数的平均数。

参数
::::::::::
   - **x** (Tensor) - 输入的 Tensor，数据类型为：float16、bfloat16、float32、float64、int32、int64。
   - **axis** (None|int|list|tuple，可选) - 指定对 ``x`` 进行计算的轴。``axis`` 可以是 int 或者 int 元素的列表。``axis`` 值应该在范围[-D， D)内，D 是 ``x`` 的维度。如果 ``axis`` 或者其中的元素值小于 0，则等价于 :math:`axis + D`。如果 ``axis`` 是 None，则对 ``x`` 的全部元素计算中位数。默认值为 None。
   - **keepdim** (bool，可选) - 是否在输出 Tensor 中保留减小的维度。如果 ``keepdim`` 为 True，则输出 Tensor 和 ``x`` 具有相同的维度(减少的维度除外，减少的维度的大小为 1)。否则，输出 Tensor 的形状会在 ``axis`` 上进行 squeeze 操作。默认值为 False
   - **mode** (str，可选) - 当输入 Tensor ``x`` 在 ``axis`` 轴上有偶数个非 NaN 元素时，可选择按照中间两个非 NaN 元素的平均值或最小值确定中位数。可选的值是 'avg' 或 'min'。默认为 'avg'。
   - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::
    Tensor 或 (Tensor, Tensor)。 若 ``mode == 'min'`` 且 ``axis`` 是 int 类型，结果返回一个元组：(非 NaN 中位数，对应下标)；否则只返回一个 Tensor （非 NaN 中位数）。

代码示例
::::::::::

COPY-FROM: paddle.nanmedian
