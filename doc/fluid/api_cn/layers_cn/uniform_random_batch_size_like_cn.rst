.. _cn_api_fluid_layers_uniform_random_batch_size_like:

uniform_random_batch_size_like
-------------------------------

.. py:function:: paddle.fluid.layers.uniform_random_batch_size_like(input, shape, dtype='float32', input_dim_idx=0, output_dim_idx=0, min=-1.0, max=1.0, seed=0)

uniform_random_batch_size_like算子。

此算子使用与输入张量（Tensor）相同的batch_size初始化张量（Tensor），并使用从均匀分布中采样的随机值。

参数：
        - **input** （Variable）- 其input_dim_idx'th维度指定batch_size的张量（Tensor）。
        - **shape** （元组|列表）- 输出的形状。
        - **input_dim_idx** （Int）- 默认值0.输入批量大小维度的索引。
        - **output_dim_idx** （Int）- 默认值0.输出批量大小维度的索引。
        - **min** （Float）- （默认 1.0）均匀随机的最小值。
        - **max** （Float）- （默认 1.0）均匀随机的最大值。
        - **seed** （Int）- （int，default 0）用于生成样本的随机种子。0表示使用系统生成的种子。注意如果seed不为0，则此算子将始终每次生成相同的随机数。
        - **dtype** （np.dtype | core.VarDesc.VarType | str） - 数据类型：float32，float_16，int等。

返回:        指定形状的张量（Tensor）将使用指定值填充。

返回类型:        Variable


**代码示例：**

.. code-block:: python

    import paddle.fluid as fluid
    import paddle.fluid.layers as layers

    input = fluid.layers.data(name="input", shape=[13, 11], dtype='float32')
    out = fluid.layers.uniform_random_batch_size_like(input, [-1, 11])





