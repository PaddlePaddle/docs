.. _cn_api_fluid_layers_reduce_prod:

reduce_prod
-------------------------------

.. py:function:: paddle.fluid.layers.reduce_prod(input, dim=None, keep_dim=False, name=None)




该OP是对指定维度上的Tensor元素进行求乘积运算，并输出相应的计算结果。

参数
::::::::::::

          - **input** （Variable）- 输入变量为多维Tensor或LoDTensor，支持数据类型为float32，float64，int32，int64。
          - **dim** （int|list|tuple，可选）- 求乘积运算的维度。如果为None，则计算所有元素的乘积并返回包含单个元素的Tensor变量，否则必须在 :math:`[−rank(input),rank(input)]` 范围内。如果 :math:`dim [i] <0`，则维度将变为 :math:`rank+dim[i]`，默认值为None。
          - **keep_dim** （bool）- 是否在输出Tensor中保留减小的维度。如 keep_dim 为true，否则结果Tensor的维度将比输入Tensor小，默认值为False。
          - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
  在指定dim上进行求乘积运算的Tensor，数据类型和输入数据类型一致。

返回类型
::::::::::::
  变量（Variable）

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.reduce_prod