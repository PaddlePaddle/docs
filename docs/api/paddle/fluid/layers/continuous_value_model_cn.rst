.. _cn_api_fluid_layers_continuous_value_model:

continuous_value_model
-------------------------------

.. py:function:: paddle.fluid.layers.continuous_value_model(input, cvm, use_cvm=True)




**注意：该 OP 仅支持在 CPU 运行。**

该 OP 在 CTR 项目中，用于去除或处理 ``input`` 中的展示和点击值。

输入 ``input`` 是一个含展示（show）和点击（click）的词向量，其形状为 :math:`[N, D]` （N 为 batch 大小，D 为 `2 + 嵌入维度` ），show 和 click 占据词向量 D 的前两维。如果 ``use_cvm=True``，它会计算 :math:`log(show)` 和 :math:`log(click)`，输出的形状为 :math:`[N, D]`。如果 ``use_cvm=False``，它会从输入 ``input`` 中移除 show 和 click，输出的形状为 :math:`[N, D - 2]` 。 ``cvm`` 为 show 和 click 信息，维度为 :math:`[N, 2]` 。

参数
::::::::::::

    - **input** (Variable) - cvm 操作的输入 Tensor。维度为 :math:`[N, D]` 的 2-D LoDTensor。 N 为 batch 大小，D 为 `2 + 嵌入维度` ， `lod level = 1` 。
    - **cvm** (Variable) - cvm 操作的展示和点击 Tensor。维度为 :math:`[N, 2]` 的 2-D Tensor。 N 为 batch 大小，2 为展示和点击值。
    - **use_cvm** (bool) - 是否使用展示和点击信息。如果使用，输出维度和输入相等，对 ``input`` 中的展示和点击值取 log；如果不使用，输出维度为输入减 2（移除展示和点击值)。

返回
::::::::::::
Variable(LoDTensor)变量，:math:`[N, M]` 的 2-D LoDTensor。如果 ``use_cvm=True`` ，M 等于输入的维度 D，否则 M 等于 `D - 2` 。

返回类型
::::::::::::
变量（Variable），数据类型与 ``input`` 一致。

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.continuous_value_model
