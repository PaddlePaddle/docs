.. _cn_api_fluid_layers_fill_constant:

fill_constant
-------------------------------

.. py:function:: paddle.fluid.layers.fill_constant(shape,dtype,value,force_cpu=False,out=None)

该OP创建一个形状为shape并且数据类型为dtype的LoDTensor或者SelectedRows，同时用 ``value`` 中提供的常量初始化该LoDTensor或者SelectedRows。

创建的LoDTensor或者SelectedRows的stop_gradient属性默认为True。

参数：
    - **shape** (tuple|list)- 创建LoDTensor或者SelectedRows的形状。
    - **dtype** (np.dtype|core.VarDesc.VarType|str)- 创建LoDTensor或者SelectedRows的数据类型，支持数据类型为float16， float32， float64， int32， int64。
    - **value** (float|int)- 用于初始化输出LoDTensor或者SelectedRows的常量数据的值。
    - **force_cpu** (bool)- 用于标志LoDTensor或者SelectedRows是否创建在CPU上，默认值为False，若设为true,则数据必须在CPU上。
    - **out** (Variable，可选)- 用于存储创建的LoDTensor或者SelectedRows，可以是程序中已经创建的任何Variable。默认值为None，此时将创建新的Variable来保存输出结果。
   

返回： 根据shape和dtype创建的LoDTensor或者SelectedRows。

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    data1 = fluid.layers.fill_constant(shape=[1], value=0, dtype='int64') #data1=[0]
    data2 = fluid.layers.fill_constant(shape=[1], value=5, dtype='int64', out=data1) #data1=[5] data2=[5]
