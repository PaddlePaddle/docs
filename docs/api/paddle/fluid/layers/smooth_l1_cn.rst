.. _cn_api_fluid_layers_smooth_l1:

smooth_l1
-------------------------------

.. py:function:: paddle.fluid.layers.smooth_l1(x, y, inside_weight=None, outside_weight=None, sigma=None)




该layer计算变量 ``x`` 和 ``y`` 的smooth L1 loss，它以 ``x`` 和 ``y`` 的第一维大小作为批处理大小。对于每个实例，按元素计算smooth L1 loss，然后计算所有loss。输出变量的形状是[batch_size, 1]


参数
::::::::::::

        - **x** (Tensor|LoDTensor) - 数据类型为float32，rank至少为2的张量。smooth L1损失函数的输入，shape为[batch_size, dim1，…，dimN]。
        - **y** (Tensor|LoDTensor) - 数据类型为float32，rank至少为2的张量。与 ``x`` shape相同的目标值。
        - **inside_weight** (Tensor|None) - 数据类型为float32，rank至少为2的张量。这个输入是可选的，与x的shape应该相同。如果给定，``(x - y)`` 的结果将乘以这个张量元素。
        - **outside_weight** (Tensor|None) - 数据类型为float32，一个rank至少为2的张量。这个输入是可选的，它的shape应该与 ``x`` 相同。smooth L1 loss的输出会乘以这个张量。
        - **sigma** (float|NoneType) - smooth L1 loss layer的超参数。标量，默认值为1.0。

返回
::::::::::::
 smooth L1损失的输出值，shape为 [batch_size, 1]

返回类型
::::::::::::
Variable（Tensor），数据类型为float32的Tensor。

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.smooth_l1