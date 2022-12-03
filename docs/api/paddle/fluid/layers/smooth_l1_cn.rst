.. _cn_api_fluid_layers_smooth_l1:

smooth_l1
-------------------------------

.. py:function:: paddle.fluid.layers.smooth_l1(x, y, inside_weight=None, outside_weight=None, sigma=None)




该 layer 计算变量 ``x`` 和 ``y`` 的 smooth L1 loss，它以 ``x`` 和 ``y`` 的第一维大小作为批处理大小。对于每个实例，按元素计算 smooth L1 loss，然后计算所有 loss。输出变量的形状是[batch_size, 1]


参数
::::::::::::

        - **x** (Tensor|LoDTensor) - 数据类型为 float32，rank 至少为 2 的 Tensor。smooth L1 损失函数的输入，shape 为[batch_size, dim1，…，dimN]。
        - **y** (Tensor|LoDTensor) - 数据类型为 float32，rank 至少为 2 的 Tensor。与 ``x`` shape 相同的目标值。
        - **inside_weight** (Tensor|None) - 数据类型为 float32，rank 至少为 2 的 Tensor。这个输入是可选的，与 x 的 shape 应该相同。如果给定，``(x - y)`` 的结果将乘以这个 Tensor 元素。
        - **outside_weight** (Tensor|None) - 数据类型为 float32，一个 rank 至少为 2 的 Tensor。这个输入是可选的，它的 shape 应该与 ``x`` 相同。smooth L1 loss 的输出会乘以这个 Tensor。
        - **sigma** (float|NoneType) - smooth L1 loss layer 的超参数。标量，默认值为 1.0。

返回
::::::::::::
 smooth L1 损失的输出值，shape 为 [batch_size, 1]

返回类型
::::::::::::
Variable（Tensor），数据类型为 float32 的 Tensor。

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.smooth_l1
