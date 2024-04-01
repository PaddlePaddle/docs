.. _cn_api_paddle_nanquantile:

nanquantile
-------------------------------

.. py:function:: paddle.nanquantile(x, q, axis=None, keepdim=False, interpolation='linear', name=None)

沿给定的轴 ``axis`` 计算 ``x`` 中元素的分位数, 忽略元素中的 ``NaN`` 。

参数
::::::::::
    - **x** (Tensor) - 输入的 Tensor，数据类型为：float32、float64。
    - **q** (int|float|list|Tensor) - 待计算的分位数，需要在符合取值范围[0, 1]。如果 ``q`` 是 List 或者 1-D Tensor，其中的每一个 q 分位数都会被计算，并且输出的首维大小与列表或 Tensor 中元素的数量相同。如果 ``q`` 是 0-D Tensor ，则会被当作 float 和 int 对待。
    - **axis** (int|list，可选) - 指定对 ``x`` 进行计算的轴。``axis`` 可以是 int 或内部元素为 int 类型的 list。``axis`` 值应该在范围[-D, D)内，D 是 ``x`` 的维度。如果 ``axis`` 或者其中的元素值小于 0，则等价于 :math:`axis + D`。如果 ``axis`` 是 list，对给定的轴上的所有元素计算分位数。如果 ``axis`` 是 None，则对 ``x`` 的全部元素计算分位数。默认值为 None。
    - **keepdim** (bool，可选) - 是否在输出 Tensor 中保留减小的维度。如果 ``keepdim`` 为 True，则输出 Tensor 和 ``x`` 具有相同的维度(减少的维度除外，减少的维度的大小为 1)。否则，输出 Tensor 的形状会在 ``axis`` 上进行 squeeze 操作。默认值为 False。
    - **interpolation** (str，可选) - 计算分位数的插值方法，可以是 ``linear`` , ``lower`` , ``higher`` , ``nearest`` 或者 ``midpoint`` 。 默认值为 ``linear`` 。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::
    ``Tensor``，沿着 ``axis`` 进行分位数计算的结果。与输入数据类型一致。

代码示例
::::::::::

COPY-FROM: paddle.nanquantile
