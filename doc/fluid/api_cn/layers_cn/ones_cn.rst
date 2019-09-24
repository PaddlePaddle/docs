.. _cn_api_fluid_layers_ones:

ones
-------------------------------

.. py:function:: paddle.fluid.layers.ones(shape,dtype,force_cpu=False)

**ones**

该OP创建形状为 ``shape`` 、数据类型为 ``dtype`` 且值全为1的Tensor，该OP会将stop_gradient设置为True，即停止梯度更新。

参数：
    - **shape** (tuple|list) - 输出Tensor的形状。
    - **dtype** (np.dtype|core.VarDesc.VarType|str) - 输出Tenor的数据类型。
    - **force_cpu** (bool) – 是否强制将输出Tensor写入CPU内存。如果 ``force_cpu`` 为False，则将输出Tensor写入当前所在运算设备的内存，默认为False。

返回：值全为1的Tensor

返回类型：Variable

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    data = fluid.layers.ones(shape=[2, 4], dtype='float32') # [[1., 1., 1., 1.], [1., 1., 1., 1.]]
