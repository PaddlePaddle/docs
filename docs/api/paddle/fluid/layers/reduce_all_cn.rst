.. _cn_api_fluid_layers_reduce_all:

reduce_all
-------------------------------

.. py:function:: paddle.fluid.layers.reduce_all(input, dim=None, keep_dim=False, name=None)




该OP是对指定维度上的Tensor元素进行与逻辑（&）计算，并输出相应的计算结果。

参数
::::::::::::

    - **input** （Variable）— 输入变量为多维Tensor或LoDTensor，数据类型需要为bool类型。
    - **dim** （list | int，可选）— 与逻辑运算的维度。如果为None，则计算所有元素的与逻辑并返回包含单个元素的Tensor变量，否则必须在 :math:`[−rank(input),rank(input))` 范围内。如果 :math:`dim [i] <0`，则维度将减小为 :math:`rank+dim[i]`。默认值为None。
    - **keep_dim** （bool）— 是否在输出Tensor中保留减小的维度。如 keep_dim 为true，否则结果张量的维度将比输入张量小，默认值为False。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
在指定dim上进行与逻辑计算的Tensor，数据类型为bool类型。

返回类型
::::::::::::
Variable，数据类型为bool类型。

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.reduce_all