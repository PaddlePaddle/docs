.. _cn_api_fluid_layers_gaussian_random_batch_size_like:

gaussian_random_batch_size_like
-------------------------------

.. py:function:: paddle.fluid.layers.gaussian_random_batch_size_like(input, shape, input_dim_idx=0, output_dim_idx=0, mean=0.0, std=1.0, seed=0, dtype='float32')

使用高斯随机发生器初始化张量。高斯分布的默认均值（mean）为0，默认标准差（std）为 1 。用户可以通过输入参数设置 mean 和 std 。

参数：
        - **input** （Variable）- 其 input_dim_idx'th 维度指定 batch_size 的张量（Tensor）。
        - **shape** （tuple|list）- 输出的形状。
        - **input_dim_idx** （Int）- （默认值0）输入批量大小维度的索引。
        - **output_dim_idx** （Int）- （默认值0）输出批量大小维度的索引。
        - **mean** （float）- （默认值 0.0）高斯分布的平均值（或中心值）。
        - **std** （float）- （默认值 1.0）高斯分布的标准差（std或spread）。
        - **seed** （int）- （默认值为 0）用于随机数发生器的随机种子。0表示使用系统生成的种子。请注意，如果seed不为0，则此算子每次将始终生成相同的随机数。
        - **dtype** （np.dtype | core.VarDesc.VarType | str）- 输出数据的类型，float32、float_16、int 等。

返回：指定形状的张量，由从高斯分布抽样产生的随机数所填充。

返回类型：Variable



**代码示例：**

.. code-block:: python

    import paddle.fluid as fluid
    input = fluid.layers.data(name="input", shape=[13, 11], dtype='float32')

    out = fluid.layers.gaussian_random_batch_size_like(
        input, shape=[-1, 11], mean=1.0, std=2.0)






