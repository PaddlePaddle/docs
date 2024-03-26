.. _cn_api_paddle_incubate_nn_FusedLinear:

FusedLinear
-------------------------------

.. py:class:: paddle.incubate.nn.FusedLinear(in_features, out_features, weight_attr=None, bias_attr=None, transpose_weight=False, name=None)

FusedLinear 层只接受一个 Tensor 作为输入，形状为 :math:`[batch\_size, *, in\_features]`，其中 :math:`*` 表示可以为任意个额外的维度。
该层可以计算输入 Tensor 与权重矩阵 :math:`W` 的乘积，然后生成形状为 :math:`[batch\_size, *, out\_features]` 的输出 Tensor。
如果 :math:`bias\_attr` 不是 False，则将创建一个偏置参数并将其添加到输出中。此方法要求 CUDA 版本不低于 11.6。

参数
::::::::::::
- **in_features** (int) – 线性变换层输入单元的数目。
- **out_features** (int) – 线性变换层输出单元的数目。
- **weight_attr** (ParamAttr，可选) - 指定权重参数的属性。默认值为 None，权重将初始化为零。具体用法请参见 :ref:`cn_api_paddle_ParamAttr` 。
- **transpose_weight** (bool) - 在乘积运算前是否转置 `weight` 张量。
- **bias_attr** (ParamAttr|bool，可选) – 指定偏置参数的属性。:math:`bias\_attr` 为 bool 类型且设置为 False 时，表示不会为该层添加偏置。:math:`bias\_attr` 如果设置为 True 或者 None，则表示使用默认的偏置参数属性，将偏置参数初始化为 0。具体用法请参见 :ref:`cn_api_paddle_ParamAttr`。默认值为 None。
- **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

属性
:::::::::

weight
'''''''''

本层的可学习参数，类型为 ``Parameter`` 。

bias
'''''''''

本层的可学习偏置，类型为 ``Parameter`` 。

形状
:::::::::

- 输入：形状为 :math:`[batch\_size, *, in\_features]` 的多维 Tensor。
- 输出：形状为 :math:`[batch\_size, *, out\_features]` 的多维 Tensor。

代码示例
:::::::::

COPY-FROM: paddle.incubate.nn.FusedLinear
