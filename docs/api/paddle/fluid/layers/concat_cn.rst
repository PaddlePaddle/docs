.. _cn_api_fluid_layers_concat:

concat
-------------------------------

.. py:function:: paddle.fluid.layers.concat(input, axis=0, name=None)


该 OP 对输入沿 ``axis`` 轴进行联结，返回一个新的 Tensor。该 API 不支持输入 0-D Tensor，因为其必须指定沿着某个轴拼接，而 0-D Tensor 没有轴。

参数
::::::::::::

    - **input** (list|tuple|Tensor) - 待联结的 Tensor list，Tensor tuple 或者 Tensor，支持的数据类型为：bool、float16、 float32、float64、int32、int64。 ``input`` 中所有 Tensor 的数据类型必须一致。
    - **axis** (int|Tensor，可选) - 指定对输入 Tensor 进行运算的轴，可以是整数或者形状为[]的 0-D Tensor，数据类型为 int32 或者 int64。 ``axis`` 的有效范围是[-R, R)，R 是输入 ``input`` 中 Tensor 的维度，``axis`` 为负值时与 :math:`axis + R` 等价。默认值为 0。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
联结后的 ``Tensor``，数据类型和 ``input`` 中的 Tensor 相同。

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.concat
