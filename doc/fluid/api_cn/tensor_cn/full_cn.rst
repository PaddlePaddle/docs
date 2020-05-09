.. _cn_api_tensor_full:

full
-------------------------------

.. py:function:: paddle.full(shape, fill_value, out=None, dtype=None, device=None, stop_gradient=True, name=None)

该OP创建一个和具有相同的形状和数据类型的Tensor，其中元素值均为fill_value。

参数：
    - **shape** (list|tuple|Variable) – 指定创建Tensor的形状(shape)。
    - **fill_value** (bool|float16|float32|int32|int64|Variable) - 用于初始化输出Tensor的常量数据的值。默认为0。注意：该参数不可超过输出变量数据类型的表示范围。
    - **out** (Variable，可选) - 输出Tensor。如果为None，则创建一个新的Tensor作为输出Tensor，默认值为None。
    - **dtype** （np.dtype|core.VarDesc.VarType|str， 可选）- 输出变量的数据类型。若参数为空，则输出变量的数据类型和输入变量相同，默认值为None。
    - **device** (str，可选) – 选择在哪个设备运行该操作，可选值包括None，'cpu'和'gpu'。如果 ``device`` 为None，则将选择运行Paddle程序的设备，默认为None。
    - **stop_gradient** (bool，可选) – 是否从此 Variable 开始，之前的相关部分都停止梯度计算，默认为True。
    - **name** （str，可选）- 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。
    
返回：返回一个存储结果的Tensor。

返回类型：Variable

抛出异常：
    - ``TypeError`` - 如果 ``dtype`` 的类型不是bool, float16, float32, float64, int32, int64其中之一。
    - ``TypeError`` - 如果 ``out`` 的元素的类型不是Variable。
    - ``TypeError`` - 如果 ``shape`` 的类型不是list或tuple或Varibable。

**代码示例**：

.. code-block:: python

    import paddle

    data1 = paddle.full(shape=[2,1], fill_value=0, dtype='int64') # data1=[[0],[0]]
    data2 = paddle.full(shape=[2,1], fill_value=5, dtype='int64', device='gpu') # data2=[[5],[5]]

    # attr shape is a list which contains Variable Tensor.
    positive_2 = paddle.fill_constant([1], "int32", 2)
    data3 = paddle.full(shape=[1, positive_2], dtype='float32', fill_value=1.5) # data3=[1.5, 1.5]

    # attr shape is an Variable Tensor.
    shape = paddle.fill_constant([1,2], "int32", 2) # shape=[2,2]
    data4 = paddle.full(shape=shape, dtype='bool', fill_value=True) # data4=[[True,True],[True,True]]
  
    # attr value is an Variable Tensor.
    val = paddle.fill_constant([1], "float32", 2.0) # val=[2.0]
    data5 = paddle.full(shape=[2,1], fill_value=val, dtype='float32') #data5=[[2.0],[2.0]]
