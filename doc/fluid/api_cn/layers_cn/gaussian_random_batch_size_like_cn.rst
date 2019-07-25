.. _cn_api_fluid_layers_gaussian_random_batch_size_like:

gaussian_random_batch_size_like
-------------------------------

.. py:function:: paddle.fluid.layers.gaussian_random_batch_size_like(input, shape, input_dim_idx=0, output_dim_idx=0, mean=0.0, std=1.0, seed=0, dtype='float32')

用于使用高斯随机发生器初始化张量。分布的defalut均值为0.并且分布的defalut标准差（std）为1.用户可以通过输入参数设置mean和std。

参数：
        - **input** （Variable）- 其input_dim_idx'th维度指定batch_size的张量（Tensor）。
        - **shape** （元组|列表）- 输出的形状。
        - **input_dim_idx** （Int）- 默认值0.输入批量大小维度的索引。
        - **output_dim_idx** （Int）- 默认值0.输出批量大小维度的索引。
        - **mean** （Float）- （默认值0.0）高斯分布的平均值（或中心值）。
        - **std** （Float）- （默认值 1.0）高斯分布的标准差（std或spread）。
        - **seed** （Int）- （默认为0）用于随机数引擎的随机种子。0表示使用系统生成的种子。请注意，如果seed不为0，则此算子将始终每次生成相同的随机数。
        - **dtype** （np.dtype | core.VarDesc.VarType | str）- 输出数据的类型为float32，float_16，int等。

返回：        指定形状的张量将使用指定值填充。

返回类型：        输出（Variable）。



**代码示例：**

.. code-block:: python

    import paddle.fluid as fluid
    input = fluid.layers.data(name="input", shape=[13, 11], dtype='float32')

    out = fluid.layers.gaussian_random_batch_size_like(
        input, shape=[-1, 11], mean=1.0, std=2.0)






