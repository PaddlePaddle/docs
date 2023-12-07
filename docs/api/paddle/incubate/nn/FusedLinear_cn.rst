.. _cn_api_paddle_incubate_nn_FusedLinear:

FusedLinear
-------------------------------

.. py:class:: paddle.incubate.nn.FusedLinear(in_features, out_features, weight_attr=None, bias_attr=None, transpose_weight=False, name=None)

线性层仅接受一个多维张量作为输入，其形状为 :math:`[batch\_size, *, in\_features]`，其中 :math:`*` 表示任意数量的额外维度。

它将输入张量与权重（形状为 :math:`[in\_features, out\_features]` 的二维张量）相乘，产生形状为 :math:`[batch\_size, *, out\_features]` 的输出张量。

如果 :math:`bias\_attr` 不为 False，则将创建偏置（形状为 :math:`[out\_features]` 的一维张量）并加到输出上。

参数
::::::::::::
    - **in_features** (int) - 输入单元的数量。
    - **out_features** (int) - 输出单元的数量。
    - **weight_attr** (ParamAttr，可选) - 指定权重参数属性的对象。默认值：``None``，表示使用默认的权重参数属性，即使用 0 进行初始化。具体用法请参见 :ref:`cn_api_paddle_ParamAttr` 。
    - **transpose_weight** (bool) - 在乘法前是否转置 `weight` 张量。
    - **bias_attr** (ParamAttr|bool，可选) - 指定偏置参数属性的对象。如果该参数值是 ``ParamAttr``，则使用 ``ParamAttr``。如果该参数为 ``bool`` 类型，只支持为 ``False``，表示没有偏置参数。默认值为 None，表示使用默认的偏置参数属性。具体用法请参见 :ref:`cn_api_paddle_ParamAttr` 。
    - **name** (str，可选) - 通常用户无需设置此参数。有关详细信息，请参阅 :ref:`api_guide_Name`。

属性
::::::::::::
    - **weight** (Parameter) - 本层的可学习权重。
    - **bias** (Parameter) - 本层的可学习偏置。

形状
::::::::::::
    - 输入：形状为 :math:`[batch\_size, *, in\_features]` 的多维张量。
    - 输出：形状为 :math:`[batch\_size, *, out\_features]` 的多维张量。

代码示例
:::::::::

COPY-FROM: paddle.incubate.nn.FusedLinear

forward(input)
::::::::::::
定义每次调用时执行的计算。应由所有子类覆盖。

参数
::::::::::::
    - **inputs** (tuple) -  解压缩的元组参数。
    - **kwargs** (dict) -  解压缩的字典参数。
