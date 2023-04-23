.. _cn_api_fluid_layers_reduce_min:

reduce_min
-------------------------------

.. py:function:: paddle.fluid.layers.reduce_min(input, dim=None, keep_dim=False, name=None)




该 OP 是对指定维度上的 Tensor 元素求最小值运算，并输出相应的计算结果。

参数
::::::::::::

          - **input** （Variable）- 输入变量为多维 Tensor，支持数据类型为 float32，float64，int32，int64。
          - **dim** （list | int，可选）- 求最小值运算的维度。如果为 None，则计算所有元素的最小值并返回包含单个元素的 Tensor 变量，否则必须在 :math:`[−rank(input),rank(input)]` 范围内。如果 :math:`dim [i] <0`，则维度将变为 :math:`rank+dim[i]`，默认值为 None。
          - **keep_dim** （bool）- 是否在输出 Tensor 中保留减小的维度。如 keep_dim 为 true，否则结果 Tensor 的维度将比输入 Tensor 小，默认值为 False。
          - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
  在指定 dim 上进行求最小值运算的 Tensor，数据类型和输入数据类型一致。

返回类型
::::::::::::
  变量（Variable）

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.reduce_min
