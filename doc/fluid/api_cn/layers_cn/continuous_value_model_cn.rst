.. _cn_api_fluid_layers_continuous_value_model:

continuous_value_model
-------------------------------

.. py:function:: paddle.fluid.layers.continuous_value_model(input, cvm, use_cvm=True)

**continuous_value_model层**

现在，continuous value model(cvm)仅考虑CTR项目中的展示和点击值。我们假设输入是一个含cvm_feature的词向量，其形状为[N * D]（D为2 + 嵌入维度）。如果use_cvm=True，它会计算log(cvm_feature)，且输出的形状为[N * D]。如果use_cvm=False，它会从输入中移除cvm_feature，且输出的形状为[N * (D - 2)]。
    
该层接受一个名为input的张量，嵌入后成为ID层(lod level为1)， cvm为一个show_click info。

参数：
    - **input** (Variable)-一个N x D的二维LodTensor， N为batch size， D为2 + 嵌入维度， lod level = 1。
    - **cvm** (Variable)-一个N x 2的二维Tensor， N为batch size，2为展示和点击值。
    - **use_cvm** (bool)-分使用/不使用cvm两种情况。如果使用cvm，输出维度和输入相等；如果不使用cvm，输出维度为输入-2（移除展示和点击值)。（cvm op是一个自定义的op，其输入是一个含embed_with_cvm默认值的序列，因此我们需要一个名为cvm的op来决定是否使用cvm。）

返回：变量，一个N x D的二维LodTensor，如果使用cvm，D等于输入的维度，否则D等于输入的维度-2。

返回类型：变量（Variable）

**代码示例**:

.. code-block:: python

    import paddle.fluid as fluid
    input = fluid.layers.data(name="input", shape=[-1, 1], lod_level=1, append_batch_size=False, dtype="int64")#, stop_gradient=False)
    label = fluid.layers.data(name="label", shape=[-1, 1], append_batch_size=False, dtype="int64")
    embed = fluid.layers.embedding(
                            input=input,
                            size=[100, 11],
                            dtype='float32')
    ones = fluid.layers.fill_constant_batch_size_like(input=label, shape=[-1, 1], dtype="int64", value=1)
    show_clk = fluid.layers.cast(fluid.layers.concat([ones, label], axis=1), dtype='float32')
    show_clk.stop_gradient = True
    input_with_cvm = fluid.layers.continuous_value_model(embed, show_clk, True)







