.. _cn_api_paddle_var:

var
-------------------------------

.. py:function:: paddle.var(x, axis=None, unbiased=True, keepdim=False, name=None)

沿给定的轴 ``axis`` 计算 ``x`` 中元素的方差。

参数
::::::::::
   - **x** (Tensor) - 输入的 Tensor，数据类型为：float16、float32、float64。
   - **axis** (int|list|tuple，可选) - 指定对 ``x`` 进行计算的轴。``axis`` 可以是 int、list(int)、tuple(int)。

      - 如果 ``axis`` 包含多个维度，则沿着 ``axis`` 中的所有轴进行计算。``axis`` 或者其中的元素值应该在范围[-D, D)内，D 是 ``x`` 的维度。
      - 如果 ``axis`` 或者其中的元素值小于 0，则等价于 :math:`axis + D` 。
      - 如果 ``axis`` 是 None，则对 ``x`` 的全部元素计算方差。默认值为 None。

   - **unbiased** (bool，可选) - 是否使用无偏估计来计算方差。使用 :math:`N` 来代表在 axis 上的维度，如果 ``unbiased`` 为 True，则在计算中使用 :math:`N - 1` 作为除数。为 False 时将使用 :math:`N` 作为除数。默认值为 True。
   - **keepdim** (bool，可选) - 是否在输出 Tensor 中保留输入的维度。除非 keepdim 为 True，否则输出 Tensor 的维度将比输入 Tensor 小一维，默认值为 False。
   - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::
    ``Tensor``，沿着 ``axis`` 进行方差计算的结果，数据类型和 ``x`` 相同。

代码示例
::::::::::

COPY-FROM: paddle.var
