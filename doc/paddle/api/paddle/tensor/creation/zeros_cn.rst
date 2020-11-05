.. _cn_api_tensor_zeros:

zeros
-------------------------------

.. py:function:: paddle.zeros(shape, dtype=None, name=None)



该OP创建形状为 ``shape`` 、数据类型为 ``dtype`` 且值全为0的Tensor。

参数：
    - **shape** (tuple|list|Tensor) - 输出Tensor的形状， ``shape`` 的数据类型为int32或者int64。
    - **dtype** (np.dtype|core.VarDesc.VarType|str，可选) - 输出Tensor的数据类型，数据类型必须为bool、float16、float32、float64、int32或int64。若为None，数据类型为float32， 默认为None。
    - **name** (str, 可选) - 输出的名字。一般无需设置，默认值为None。该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` 。

返回：值全为0的Tensor，数据类型和 ``dtype`` 定义的类型一致。

抛出异常：
    - ``TypeError`` - 当 ``dtype`` 不是bool、 float16、float32、float64、int32、int64和None时。
    - ``TypeError`` - 当 ``shape`` 不是tuple、list、或者Tensor时， 当 ``shape`` 为Tensor，其数据类型不是int32或者int64时。

**代码示例**：

.. code-block:: python

    import paddle
    data = paddle.zeros(shape=[3, 2], dtype='float32') 
    # [[0. 0.]
    #  [0. 0.]
    #  [0. 0.]]
    
    data = paddle.zeros(shape=[2, 2]) 
    # [[0. 0.]
    #  [0. 0.]]
    
    # shape is a Tensor
    shape = paddle.full(shape=[2], dtype='int32', fill_value=2)
    data3 = paddle.zeros(shape=shape, dtype='int32') 
    # [[0 0]
    #  [0 0]]

