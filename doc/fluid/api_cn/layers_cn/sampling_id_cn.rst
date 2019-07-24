.. _cn_api_fluid_layers_sampling_id:

sampling_id
-------------------------------

.. py:function:: paddle.fluid.layers.sampling_id(x, min=0.0, max=1.0, seed=0, dtype='float32')

sampling_id算子。用于从输入的多项分布中对id进行采样的图层。为一个样本采样一个id。

参数：
        - **x** （Variable）- softmax的输入张量（Tensor）。2-D形状[batch_size，input_feature_dimensions]
        - **min** （Float）- 随机的最小值。（浮点数，默认为0.0）
        - **max** （Float）- 随机的最大值。（float，默认1.0）
        - **seed** （Float）- 用于随机数引擎的随机种子。0表示使用系统生成的种子。请注意，如果seed不为0，则此算子将始终每次生成相同的随机数。（int，默认为0）
        - **dtype** （np.dtype | core.VarDesc.VarType | str）- 输出数据的类型为float32，float_16，int等。

返回：       Id采样的数据张量。

返回类型：        输出（Variable）。


**代码示例：**

.. code-block:: python

    import paddle.fluid as fluid
    x = fluid.layers.data(
    name="X",
    shape=[13, 11],
    dtype='float32',
    append_batch_size=False)

    out = fluid.layers.sampling_id(x)








