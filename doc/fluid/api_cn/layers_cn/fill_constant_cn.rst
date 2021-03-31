.. _cn_api_fluid_layers_fill_constant:

fill_constant
-------------------------------

.. py:function:: paddle.fluid.layers.fill_constant(shape,dtype,value,force_cpu=False,out=None)

:alias_main: paddle.fill_constant
:alias: paddle.fill_constant,paddle.tensor.fill_constant,paddle.tensor.creation.fill_constant
:old_api: paddle.fluid.layers.fill_constant



该OP创建一个形状为shape并且数据类型为dtype的Tensor，同时用 ``value`` 中提供的常量初始化该Tensor。

创建的Tensor的stop_gradient属性默认为True。

参数：
    - **shape** (tuple|list|Variable)- 要创建的LoDTensor或者SelectedRows的形状。 数据类型为int32或int64。 如果shape是一个列表或元组，则其元素应该是形状为[1]的整数或Tensor。 如果shape是Variable，则它应该是一维Tensor。
    - **dtype** (np.dtype|core.VarDesc.VarType|str)- 创建LoDTensor或者SelectedRows的数据类型，支持数据类型为bool， float16， float32， float64， uint8， int32， int64。
    - **value** (float|int)- 用于初始化输出LoDTensor或者SelectedRows的常量数据的值。
    - **force_cpu** (bool)- 用于标志LoDTensor或者SelectedRows是否创建在CPU上，默认值为False，若设为true,则数据必须在CPU上。
    - **out** (Variable，可选)- 用于存储创建的LoDTensor或者SelectedRows，可以是程序中已经创建的任何Variable。默认值为None，此时将创建新的Variable来保存输出结果。
   

返回： 根据shape和dtype创建的Tensor。

返回类型：变量（Variable）

抛出异常：
    - :code:`TypeError`: dtype必须是bool，float16，float32，float64，uint8，int32和int64之一，输出Tensor的数据类型必须与dtype相同。
    - :code:`TypeError`: 当 `shape` 的数据类型不是list、tuple、Variable。

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    data1 = fluid.layers.fill_constant(shape=[2,1], value=0, dtype='int64') #data1=[[0],[0]]
    data2 = fluid.layers.fill_constant(shape=[2,1], value=5, dtype='int64', out=data1) 
    #data1=[[5],[5]] data2=[[5],[5]]

    # attr shape is a list which contains Variable Tensor.
    positive_2 = fluid.layers.fill_constant([1], "int32", 2)
    data3 = fluid.layers.fill_constant(shape=[1, positive_2], dtype='float32', value=1.5) # data3=[1.5, 1.5]

    # attr shape is a Variable Tensor.
    shape = fluid.layers.fill_constant([1,2], "int32", 2) # shape=[2,2]
    data4 = fluid.layers.fill_constant(shape=shape, dtype='bool', value=True) # data4=[[True,True],[True,True]]
