.. _cn_api_fluid_layers_smooth_l1:

smooth_l1
-------------------------------

.. py:function:: paddle.fluid.layers.smooth_l1(x, y, inside_weight=None, outside_weight=None, sigma=None)

该layer计算变量 ``x`` 和 ``y`` 的smooth L1 loss，它以 ``x`` 和 ``y`` 的第一维大小作为批处理大小。对于每个实例，按元素计算smooth L1 loss，然后计算所有loss。输出变量的形状是[batch_size, 1]


参数:
        - **x** (Variable) - rank至少为2的张量。输入x的smmoth L1 loss 的op，shape为[batch_size, dim1，…],dimN]。
        - **y** (Variable) - rank至少为2的张量。与 ``x`` 形状一致的的smooth L1 loss  op目标值。
        - **inside_weight** (Variable|None) - rank至少为2的张量。这个输入是可选的，与x的形状应该相同。如果给定， ``(x - y)`` 的结果将乘以这个张量元素。
        - **outside_weight** (变量|None) - 一个rank至少为2的张量。这个输入是可选的，它的形状应该与 ``x`` 相同。如果给定，那么 smooth L1 loss 就会乘以这个张量元素。
        - **sigma** (float|None) - smooth L1 loss layer的超参数。标量，默认值为1.0。

返回： smooth L1 loss, shape为 [batch_size, 1]

返回类型:  Variable

**代码示例**

..  code-block:: python

    import paddle.fluid as fluid
    data = fluid.layers.data(name='data', shape=[128], dtype='float32')
    label = fluid.layers.data(
        name='label', shape=[100], dtype='float32')
    fc = fluid.layers.fc(input=data, size=100)
    out = fluid.layers.smooth_l1(x=fc, y=label)










