.. _cn_api_fluid_layers_continuous_value_model:

continuous_value_model
-------------------------------

.. py:function:: paddle.fluid.layers.continuous_value_model(input, cvm, use_cvm=True)

:alias_main: paddle.nn.functional.continuous_value_model
:alias: paddle.nn.functional.continuous_value_model,paddle.nn.functional.extension.continuous_value_model
:old_api: paddle.fluid.layers.continuous_value_model



**注意：该OP仅支持在CPU运行。**

该OP在CTR项目中，用于去除或处理 ``input`` 中的展示和点击值。

输入 ``input`` 是一个含展示（show）和点击（click）的词向量，其形状为 :math:`[N, D]` （N为batch大小，D为 `2 + 嵌入维度` ），show和click占据词向量D的前两维。如果 ``use_cvm=True`` ，它会计算 :math:`log(show)` 和 :math:`log(click)` ，输出的形状为 :math:`[N, D]` 。如果 ``use_cvm=False`` ，它会从输入 ``input`` 中移除show和click，输出的形状为 :math:`[N, D - 2]` 。 ``cvm`` 为show和click信息，维度为 :math:`[N, 2]` 。

参数：
    - **input** (Variable) - cvm操作的输入张量。维度为 :math:`[N, D]` 的2-D LoDTensor。 N为batch大小， D为 `2 + 嵌入维度` ， `lod level = 1` 。
    - **cvm** (Variable) - cvm操作的展示和点击张量。维度为 :math:`[N, 2]` 的2-D Tensor。 N为batch大小，2为展示和点击值。
    - **use_cvm** (bool) - 是否使用展示和点击信息。如果使用，输出维度和输入相等，对 ``input`` 中的展示和点击值取log；如果不使用，输出维度为输入减2（移除展示和点击值)。

返回：Variable(LoDTensor)变量， :math:`[N, M]` 的2-D LoDTensor。如果 ``use_cvm=True`` ，M等于输入的维度D，否则M等于 `D - 2` 。

返回类型：变量（Variable）,数据类型与 ``input`` 一致。

**代码示例**:

.. code-block:: python

    import paddle.fluid as fluid
    input = fluid.layers.data(name="input", shape=[-1, 1], lod_level=1, append_batch_size=False, dtype="int64")
    label = fluid.layers.data(name="label", shape=[-1, 1], append_batch_size=False, dtype="int64")
    embed = fluid.layers.embedding(
                            input=input,
                            size=[100, 11],
                            dtype='float32')
    label_shape = fluid.layers.shape(label)
    ones = fluid.layers.fill_constant(shape=[label_shape[0], 1], dtype="int64", value=1)
    show_clk = fluid.layers.cast(fluid.layers.concat([ones, label], axis=1), dtype='float32')
    show_clk.stop_gradient = True
    input_with_cvm = fluid.layers.continuous_value_model(embed, show_clk, True)







