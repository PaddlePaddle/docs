.. _cn_api_tensor_mean:

mean
-------------------------------

.. py:function:: paddle.mean(x, axis=None, keepdim=False, name=None)



该OP沿 ``axis`` 计算 ``x`` 的平均值。

参数
::::::::::
    - x (Tensor) - 输入的Tensor，数据类型为：float32、float64、int32.int64 。
    - axis (int|list|tuple)。

返回：值全为0的Tensor，数据类型和 ``dtype`` 定义的类型一致。



**代码示例**：

.. code-block:: python

    import paddle
    paddle.enable_imperative()  # Now we are in imperative mode
    data = paddle.zeros(shape=[3, 2], dtype='float32') 
    # [[0. 0.]
    #  [0. 0.]
    #  [0. 0.]]
    
    data = paddle.zeros(shape=[2, 2]) 
    # [[0. 0.]
    #  [0. 0.]]
    
    # shape is a Tensor
    shape = paddle.fill_constant(shape=[2], dtype='int32', value=2)
    data3 = paddle.ones(shape=shape, dtype='int32') 
    # [[0 0]
    #  [0 0]]

