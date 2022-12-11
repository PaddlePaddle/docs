.. _cn_api_paddle_tensor_amin:

amin
-------------------------------

.. py:function:: paddle.amin(x, axis=None, keepdim=False, name=None)


对指定维度上的 Tensor 元素求最小值运算，并输出相应的计算结果。

.. note::

    对输入有多个最小值的情况下，min 将梯度完整传回到最小值对应的位置，amin 会将梯度平均传回到最小值对应的位置

参数
:::::::::
   - **x** (Tensor) - Tensor，支持数据类型为 float32、float64、int32、int64，维度不超过 4 维。
   - **axis** (int|list|tuple，可选) - 求最小值运算的维度。如果为 None，则计算所有元素的最小值并返回包含单个元素的 Tensor 变量，否则必须在 :math:`[−x.ndim, x.ndim]` 范围内。如果 :math:`axis[i] < 0`，则维度将变为 :math:`x.ndim+axis[i]`，默认值为 None。
   - **keepdim** (bool，可选)- 是否在输出 Tensor 中保留减小的维度。如果 keepdim 为 False，结果 Tensor 的维度将比输入 Tensor 的小，默认值为 False。
   - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
   Tensor，在指定 axis 上进行求最小值运算的 Tensor，数据类型和输入数据类型一致。


代码示例
::::::::::
COPY-FROM: paddle.amin
