.. _cn_api_paddle_std:

std
-------------------------------

.. py:function:: paddle.std(x, axis=None, unbiased=None, keepdim=False, name=None, *, correction=1, out=None)

沿给定的轴 ``axis`` 计算 ``x`` 中元素的标准差。

本 API 支持两种调用方式：

1. **Paddle 风格**： ``paddle.std(x, axis=None, unbiased=None, keepdim=False, name=None, *, correction=1, out=None)``
   ``keepdim`` 作为位置参数传入。

2. **PyTorch 风格**： ``paddle.std(input, dim=None, *, correction=1, keepdim=False, out=None)``
   ``keepdim`` 作为仅关键字参数传入。

参数
::::::::::
    - **x** (Tensor) - 输入的 Tensor，数据类型为：float16、float32、float64。别名： ``input``。
    - **axis** (int|list|tuple，可选) - 指定对 ``x`` 进行计算的轴。``axis`` 可以是 int、list(int)、tuple(int)。别名： ``dim``。

        - 如果 ``axis`` 包含多个维度，则沿着 ``axis`` 中的所有轴进行计算。``axis`` 或者其中的元素值应该在范围[-D, D)内，D 是 ``x`` 的维度。
        - 如果 ``axis`` 或者其中的元素值小于 0，则等价于 :math:`axis + D` 。
        - 如果 ``axis`` 是 None，则对 ``x`` 的全部元素计算标准差。默认值为 None。

    - **unbiased** (bool，可选) - 是否使用无偏估计来计算标准差。使用 :math:`N` 来代表在 axis 上的维度，如果 ``unbiased`` 为 True，则在计算中使用 :math:`N - 1` 作为除数。为 False 时将使用 :math:`N` 作为除数。默认值为 None。
    - **keepdim** (bool，可选) - 是否在输出 Tensor 中保留减小的维度。如果 ``keepdim`` 为 True，则输出 Tensor 和 ``x`` 具有相同的维度(减少的维度除外，减少的维度的大小为 1)。否则，输出 Tensor 的形状会在 ``axis`` 上进行 squeeze 操作。默认值为 False。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

关键字参数
::::::::::
    - **correction** (int|float，可选) - 样本数量与样本自由度之间的差异。该参数为仅关键字参数，默认值为 1（贝塞尔校正）。若已设置 ``unbiased`` 为 True，则 ``correction`` 为 1，无需手动设置。
    - **out** (Tensor，可选) - 输出 Tensor，若不为 ``None``，计算结果将保存在该 Tensor 中，默认值为 ``None``。

返回
::::::::::
    ``Tensor``，沿着 ``axis`` 进行标准差计算的结果，数据类型和 ``x`` 相同。

代码示例
::::::::::

COPY-FROM: paddle.std
