.. _cn_api_fluid_layers_sampling_id:

sampling_id
-------------------------------

.. py:function:: paddle.fluid.layers.sampling_id(x, min=0.0, max=1.0, seed=0, dtype='float32')

该OP从输入的多项分布中进行采样。

参数：
        - **x** （Variable）- 输入Tensor。一个形如[batch_size，input_feature_dimensions]的2-D Tensor。
        - **min** （Float）- 随机的最小值。默认值为为0.0。
        - **max** （Float）- 随机的最大值。默认值为1.0。
        - **seed** （int）- 随机种子。0表示使用系统生成的种子。请注意，如果seed不为0，则此算子将始终每次生成相同的随机数。默认值为0
        - **dtype** （np.dtype | core.VarDesc.VarType | str）- 指定输出数据的类型。

返回：采样的数据Tensor

返回类型：Variable


**代码示例：**

.. code-block:: python

    import paddle.fluid as fluid
    x = fluid.layers.data(
    name="X",
    shape=[13, 11],
    dtype='float32',
    append_batch_size=False)

    out = fluid.layers.sampling_id(x)








