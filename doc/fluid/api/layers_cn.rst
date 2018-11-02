
.. _cn_api_fluid_layers:

While
>>>>>>>>>>>>

*class* paddle.fluid.layers.  While *(cond, is_test=False, name=None)*
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

该类用于实现while循环控制功能。


参数：  
		- cond (Variable) – 用于比较的条件
		- is_test (bool) – 用于表明是不是在测试阶段执行

 
**代码示例**

..  code-block:: python

  d0 = layers.data("d0", shape=[10], dtype='float32')
  data_array = layers.array_write(x=d0, i=i)
  array_len = layers.fill_constant(shape=[1],dtype='int64', value=3)
  cond = layers.less_than(x=i, y=array_len)
  while_op = layers.While(cond=cond)
  with while_op.block():
      d = layers.array_read(array=data_array, i=i)
      i = layers.increment(x=i, in_place=True)
      layers.array_write(result, i=i, array=d)
      layers.less_than(x=i, y=array_len, cond=cond)


Switch
>>>>>>>>>>>>>>>>>>>>
*class* paddle.fluid.layers.  Switch *(name=None)*
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Switch类实现的功能十分类似if-elif-else。它可以在学习率调度器(learning rate scheduler)中调整学习率。
:: 
  语义上，
      1. switch控制流挨个检查cases
      2. 各个case的条件是一个布尔值(boolean)，它是一个标量(scalar)变量
      3. 它将执行第一个匹配的case后面的分支，如果没有匹配的case，但若存在一个default case,则会执行default case后面的语句
      4. 一旦匹配了一个case,它降会执行这个case所对应的分支，且仅此分支。

**代码示例**

..  code-block:: python
    
    lr = fluid.layers.tensor.create_global_var(
        shape=[1],
        value=0.0,
        dtype='float32',
        persistable=True,
        name="learning_rate")
    one_var = tensor.fill_constant(
        shape=[1], dtype='float32', value=1.0)
    two_var = tensor.fill_constant(
        shape=[1], dtype='float32', value=2.0)

    with fluid.layers.control_flow.Switch() as switch:
        with switch.case(global_step == zero_var):
            fluid.layers.tensor.assign(input=one_var, output=lr)
        with switch.default():
            fluid.layers.tensor.assign(input=two_var, output=lr)
 
``case(condition)``
""""""""""""""""""""""""""""""""""
为该condition（情况，条件）建立新的block（块）。
  
  
``default()``
""""""""""""""""""""""""""""""""""""""
为该switch建立default case。
  
  
increment
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
  
paddle.fluid.layers.  increment(x, value=1.0, in_place=True)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
   
该函数为x中的每一个值增加 ``value`` 大小, ``value`` 即函数中待传入的参数。该函数默认直接在原变量x上进行运算。
  
参数:
    - x (Variable|list) – 含有输入值的张量(tensor)
    - value (float) – 需要增加在x变量上的值
    - in_place (bool) – 是否在x变量本身进行增加操作，而非返回其增加后的一个副本本身不改变。默认为True, 即在其本身进行操作。

返回： 每个元素增加后的对象
返回类型：变量(variable)

**代码示例**

..  code-block:: python
  
    data = fluid.layers.data(name='data', shape=[32, 32], dtype='float32')
    data = fluid.layers.increment(x=data, value=3.0, in_place=True)
    
    
    
array_write
>>>>>>>>>>>>>>>>>>>>>>
paddle.fluid.layers.   array_write(x, i, array=None)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
该函数将给定的输入变量（即 ``x`` ）写入一个作为输出的 ``LOD_TENSOR_ARRAY`` 变量的某一指定位置中，
这一位置由数组下标(即 ``i`` )指明。 如果 ``LOD_TENSOR_ARRAY`` (即 ``array`` )未指定（即为None值）， 一个新的 ``LOD_TENSOR_ARRAY`` 将会被创建并作为结果返回。

参数:
    - x (Variable|list) – 待从中读取数据的输入张量(tensor)
    - i (Variable|list) – 输出结果 ``LOD_TENSOR_ARRAY`` 的下标, 该下标指向输入张量 ``x`` 写入输出数组的位置
    - array (Variable|list) – 会被输入张量 ``x`` 写入的输出结果 ``LOD_TENSOR_ARRAY`` 。如果该项值为None, If this parameter is NONE, 一个新的 ``LOD_TENSOR_ARRAY`` 将会被创建并作为结果返回
 
返回:	输入张量 ``x`` 所写入的输出结果 ``LOD_TENSOR_ARRAY``  
返回类型:	变量（Variable）

**代码示例**

..  code-block:: python

  tmp = fluid.layers.zeros(shape=[10], dtype='int32')
  i = fluid.layers.fill_constant(shape=[1], dtype='int64', value=10)
  arr = layers.array_write(tmp, i=i)

