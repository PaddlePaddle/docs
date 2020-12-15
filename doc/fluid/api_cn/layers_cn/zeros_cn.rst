.. _cn_api_fluid_layers_zeros:

zeros
-------------------------------

.. py:function:: paddle.fluid.layers.zeros(shape,dtype,force_cpu=False)

该OP创建形状为 ``shape`` 、数据类型为 ``dtype`` 且值全为0的Tensor。

参数：
    - **shape** (tuple|list|Tensor) - 输出Tensor的形状， ``shape`` 的数据类型为int32或者int64。
    - **dtype** (np.dtype|str) - 输出Tensor的数据类型，数据类型必须为bool、 float16、float32、float64、int32或int64。
    - **force_cpu** (bool， 可选) - 是否强制将输出Tensor写入CPU内存。如果 ``force_cpu`` 为False，则将输出Tensor写入当前所在运算设备的内存，默认为False。

返回：值全为0的Tensor，数据类型和 ``dtype`` 定义的类型一致。

抛出异常：
    - ``TypeError`` - 当 ``dtype`` 不是bool、 float16、float32、float64、int32、int64。
    - ``TypeError`` - 当 ``shape`` 不是tuple、list、或者Tensor时。 当 ``shape`` 为Tensor，其数据类型不是int32或者int64时。

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    data = fluid.layers.zeros(shape=[3, 2], dtype='float32') # [[0., 0.], [0., 0.], [0., 0.]]
