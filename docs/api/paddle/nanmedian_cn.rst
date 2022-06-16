.. _cn_api_tensor_cn_nanmedian:

nanmedian
-------------------------------

.. py:function:: paddle.nanmedian(x, axis=None, keepdim=True, name=None)

沿给定的轴 ``axis`` 忽略NAN元素计算 ``x`` 中元素的中位数。

参数
::::::::::
   - x (Tensor) - 输入的Tensor，数据类型为：float16、float32、float64、int32、int64。
   - axis (None|int|list|tuple，可选) - 指定对 ``x`` 进行计算的轴。``axis`` 可以是int或者int元素的列表。``axis`` 值应该在范围[-D， D)内，D是 ``x`` 的维度。如果 ``axis`` 或者其中的元素值小于0，则等价于 :math:`axis + D` 。如果 ``axis`` 是None，则对 ``x`` 的全部元素计算中位数。默认值为None。
   - keepdim (bool，可选) - 是否在输出Tensor中保留减小的维度。如果 ``keepdim`` 为True，则输出Tensor和 ``x`` 具有相同的维度(减少的维度除外，减少的维度的大小为1)。否则，输出Tensor的形状会在 ``axis`` 上进行squeeze操作。默认值为True。
   - **name** (str，可选) - 具体用法请参见  :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::
    ``Tensor``，沿着 ``axis`` 忽略NAN元素进行中位数计算的结果。

代码示例
::::::::::
COPY-FROM: paddle.nanmedian:nanmedian-example

