.. _cn_api_paddle_tensor_amin:

amin
-------------------------------

.. py:function:: paddle.tensor.amin(x, axis=None, keepdim=False, name=None)

:alias_main: paddle.amin
:alias: paddle.amin,paddle.tensor.amin,paddle.tensor.math.amin

该OP是对指定维度上的Tensor元素求最小值运算，并输出相应的计算结果。

.. note::

    对输入有多个最小值的情况下，min 将梯度完整传回到最小值对应的位置，amin 会将梯度平均传回到最小值对应的位置

参数
:::::::::
   - **x** （Tensor）- Tensor，支持数据类型为float32，float64，int32，int64，维度不超过4维。
   - **axis** （list | int ，可选）- 求最小值运算的维度。如果为None，则计算所有元素的最小值并返回包含单个元素的Tensor变量，否则必须在  :math:`[−x.ndim, x.ndim]` 范围内。如果 :math:`axis[i] < 0` ，则维度将变为 :math:`x.ndim+axis[i]` ，默认值为None。
   - **keepdim** （bool）- 是否在输出Tensor中保留减小的维度。如果keepdim 为False，结果张量的维度将比输入张量的小，默认值为False。
   - **name** （str， 可选）- 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回
:::::::::
   Tensor，在指定axis上进行求最小值运算的Tensor，数据类型和输入数据类型一致。


代码示例
::::::::::
COPY-FROM: paddle.amin
