.. _cn_api_tensor_full:

full
-------------------------------

.. py:function:: paddle.full(shape, fill_value, dtype=None, name=None)



该OP创建形状大小为shape并且数据类型为dtype的Tensor，其中元素值均为 ``fill_value``。

参数：
    - **shape** (list|tuple|Tensor) – 指定创建Tensor的形状(shape), 数据类型为int32 或者int64。
    - **fill_value** (bool|float|int|Tensor) - 用于初始化输出Tensor的常量数据的值。注意：该参数不可超过输出变量数据类型的表示范围。
    - **dtype** （np.dtype|core.VarDesc.VarType|str， 可选）- 输出变量的数据类型。若为None，则输出变量的数据类型和输入变量相同，默认值为None。
    - **name** （str，可选）- 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。
    
返回：返回一个存储结果的Tensor，数据类型和dtype相同。


抛出异常：
    - ``TypeError``: - 如果 ``dtype`` 的类型不是bool, float16, float32, float64, int32, int64其中之一。
    - ``TypeError``: - 如果 ``shape`` 的类型不是list或tuple或Tensor。当 ``shape`` 是Tensor的时候，其数据类型不是int32或者int64时。

**代码示例**：

.. code-block:: python

    import paddle

    paddle.enable_imperative()  # Now we are in imperative mode
    data1 = paddle.full(shape=[2,1], fill_value=0, dtype='int64') 
    #[[0]
    # [0]]

    # attr shape is a list which contains Tensor.
    positive_3 = paddle.fill_constant([1], "int32", 2)
    data3 = paddle.full(shape=[1, positive_2], dtype='float32', fill_value=1.5)
    # [[1.5 1.5]]

    # attr shape is a Tensor.
    shape = paddle.fill_constant([2], "int32", 2)
    data4 = paddle.full(shape=shape, dtype='bool', fill_value=True) 
    # [[True True] 
    #  [True True]]
    
    # attr fill_value is a Tensor.
    val = paddle.fill_constant([1], "float32", 2.0)
    data5 = paddle.full(shape=[2,1], fill_value=val, dtype='float32') i
    # [[2.0] 
    #  [2.0]]
