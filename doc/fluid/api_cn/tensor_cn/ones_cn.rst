.. _cn_api_tensor_ones:

ones
-------------------------------

.. py:function:: paddle.ones(shape, dtype=None)

:alias_main: paddle.ones
:alias: paddle.ones,paddle.tensor.ones,paddle.tensor.creation.ones
:update_api: paddle.fluid.layers.ones




该OP创建形状为 ``shape`` 、数据类型为 ``dtype`` 且值全为1的Tensor。

参数：
    - **shape** (tuple|list|Variable) - 输出Tensor的形状，数据类型为int32或者int64。
    - **dtype** (np.dtype|core.VarDesc.VarType|str， 可选) - 输出Tensor的数据类型，数据类型必须为float16、float32、float64、int32或int64。如果dtype为None，默认数据类型为float32。
    - **name** （str，可选）- 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回：值全为1的Tensor，数据类型和 ``dtype`` 定义的类型一致。

返回类型：Variable

抛出异常：
    - ``TypeError`` - 当dtype不是float16、float32、float64、int32或int64中的一个的时候
    - ``TypeError`` - 当shape 不是tuple、list、或者Variable的时候。

**代码示例**：

.. code-block:: python

    import paddle
    
    paddle.enable_imperative()
    data1 = paddle.ones(shape=[3, 2]) 
    # [[1. 1.]
    #  [1. 1.]
    #  [1. 1.]]
    data2 = paddle.ones(shape=[2, 2], dtype='int32') 
    # [[1 1]
    #  [1 1]]
    shape = paddle.fill_constant(shape=[2], dtype='int32', value=2)
    data3 = paddle.ones(shape=shape, dtype='int32') 
    # [[1 1]
    #  [1 1]]

