===========
tensor
===========


.. _cn_api_fluid_layers_argmax:

argmax
-------------------------------

.. py:function:: paddle.fluid.layers.argmax(x,axis=0)

**argmax**

该功能计算输入张量元素中最大元素的索引，张量的元素在提供的轴上。

参数：
    - **x** (Variable)-用于计算最大元素索引的输入
    - **axis** (int)-用于计算索引的轴

返回：存储在输出中的张量

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python
    
    x = fluid.layers.data(name="x", shape=[3, 4], dtype="float32")
    out = fluid.layers.argmax(x=in, axis=0)
    out = fluid.layers.argmax(x=in, axis=-1)









.. _cn_api_fluid_layers_argmin:

argmin
-------------------------------

.. py:function:: paddle.fluid.layers.argmin(x,axis=0)

**argmin**

该功能计算输入张量元素中最小元素的索引，张量元素在提供的轴上。

参数：
    - **x** (Variable)-计算最小元素索引的输入
    - **axis** (int)-计算索引的轴

返回：存储在输出中的张量

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python
    
    x = fluid.layers.data(name="x", shape=[3, 4], dtype="float32")
    out = fluid.layers.argmin(x=in, axis=0)
    out = fluid.layers.argmin(x=in, axis=-1)









.. _cn_api_fluid_layers_argsort:

argsort
-------------------------------

.. py:function:: paddle.fluid.layers.argsort(input,axis=-1,name=None)

对输入变量沿给定轴进行排序，输出排序好的数据和相应的索引，其维度和输入相同

.. code-block:: text

    例如：
  给定 input 并指定 axis=-1

        input = [[0.15849551, 0.45865775, 0.8563702 ],
                [0.12070083, 0.28766365, 0.18776911]],

      执行argsort操作后，得到排序数据：

        out = [[0.15849551, 0.45865775, 0.8563702 ],
            [0.12070083, 0.18776911, 0.28766365]],

  根据指定axis排序后的数据indices变为:

        indices = [[0, 1, 2],
                [0, 2, 1]]

参数：
    - **input** (Variable)-用于排序的输入变量
    - **axis** (int)- 沿该参数指定的轴对输入进行排序。当axis<0,实际的轴为axis+rank(input)。默认为-1，即最后一维。
    - **name** (str|None)-（可选）该层名称。如果设为空，则自动为该层命名。

返回：一组已排序的数据变量和索引

返回类型：元组

**代码示例**：

.. code-block:: python

    x = fluid.layers.data(name="x", shape=[3, 4], dtype="float32")
    out, indices = fluid.layers.argsort(input=x, axis=0)









.. _cn_api_fluid_layers_assign:

assign
-------------------------------

.. py:function:: paddle.fluid.layers.assign(input,output=None)

该函数将输入变量复制到输出变量

参数：
    - **input** (Variable|numpy.ndarray)-源变量
    - **output** (Variable|None)-目标变量

返回：作为输出的目标变量

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    data = fluid.layers.data(name="data", shape=[3, 32, 32], dtype="float32")
    out = fluid.layers.create_tensor(dtype='float32')
    hidden = fluid.layers.fc(input=data, size=10)
    fluid.layers.assign(hidden, out)









.. _cn_api_fluid_layers_cast:

cast
-------------------------------

.. py:function:: paddle.fluid.layers.cast(x,dtype)

该层传入变量x, 并用x.dtype将x转换成dtype类型，作为输出。如果输出的dtype和输入的dtype相同，则使用cast是没有意义的，但如果真的这么做了也不会报错。

参数：
    - **x** (Variable)-转换函数的输入变量
    - **dtype** (np.dtype|core.VarDesc.VarType|str)-输出变量的数据类型

返回：转换后的输出变量

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    data = fluid.layers.data(name='x', shape=[13], dtype='float32')
    result = fluid.layers.cast(x=data, dtype='float64')









.. _cn_api_fluid_layers_concat:

concat
-------------------------------

.. py:function:: paddle.fluid.layers.concat(input,axis=0,name=None)

**Concat**

这个函数将输入连接在前面提到的轴上，并将其作为输出返回。

参数：
    - **input** (list)-将要联结的张量列表
    - **axis** (int)-数据类型为整型的轴，其上的张量将被联结
    - **name** (str|None)-该层名称（可选）。如果设为空，则自动为该层命名。

返回：输出的联结变量

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python
    
    a = fluid.layers.data(name='a', shape=[2, 13], dtype='float32')
    b = fluid.layers.data(name='b', shape=[2, 3], dtype='float32')
    c = fluid.layers.data(name='c', shape=[2, 2], dtype='float32')
    d = fluid.layers.data(name='d', shape=[2, 5], dtype='float32')
    out = fluid.layers.concat(input=[Efirst, Esecond, Ethird, Efourth])









.. _cn_api_fluid_layers_create_global_var:

create_global_var
-------------------------------

.. py:function:: paddle.fluid.layers.create_global_var(shape,value,dtype,persistable=False,force_cpu=False,name=None)

在全局块中创建一个新的带有 ``value`` 的张量。

参数：
    - **shape** (list[int])-变量的维度
    - **value** (float)-变量的值。填充新创建的变量
    - **dtype** (string)-变量的数据类型
    - **persistable** (bool)-如果是永久变量。默认：False
    - **force_cpu** (bool)-将该变量压入CPU。默认：False
    - **name** (str|None)-变量名。如果设为空，则自动创建变量名。默认：None.

返回：创建的变量

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    import paddle.fluid.layers as layers
    var = layers.create_global_var(shape=[2,3], value=1.0, dtype='float32',
                     persistable=True, force_cpu=True, name='new_var')









.. _cn_api_fluid_layers_create_parameter:

create_parameter
-------------------------------

.. py:function:: paddle.fluid.layers.create_parameter(shape,dtype,name=None,attr=None,is_bias=False,default_initializer=None)

创建一个参数。该参数是一个可学习的变量，拥有梯度并且可优化。

注：这是一个低级别的API。如果您希望自己创建新的op，这个API将非常有用，无需使用layers。

参数：
    - **shape** (list[int])-参数的维度
    - **dtype** (string)-参数的元素类型
    - **attr** (ParamAttr)-参数的属性
    - **is_bias** (bool)-当default_initializer为空，该值会对选择哪个默认初始化程序产生影响。如果is_bias为真，则使用initializer.Constant(0.0)，否则使用Xavier()。
    - **default_initializer** (Initializer)-参数的初始化程序

返回：创建的参数

**代码示例**：

.. code-block:: python

    import paddle.fluid.layers as layers
    W = fluid.layers.create_parameter(shape=[784, 200], dtype='float32')









.. _cn_api_fluid_layers_create_tensor:

create_tensor
-------------------------------

.. py:function:: paddle.fluid.layers.create_tensor(dtype,name=None,persistable=False)

创建一个变量，存储数据类型为dtype的LoDTensor。

参数：
    - **dtype** (string)-“float32”|“int32”|..., 创建张量的数据类型。
    - **name** (string)-创建张量的名称。如果未设置，则随机取一个唯一的名称。
    - **persistable** (bool)-是否将创建的张量设置为 persistable

返回：一个张量，存储着创建的张量

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    tensor = fluid.layers.create_tensor(dtype='float32')



.. _cn_api_fluid_layers_diag:

diag
-------------------------------

.. py:function:: paddle.fluid.layers.diag(diagonal)

该功能创建一个方阵，含有diagonal指定的对角线值。

参数：
    - **diagonal** (Variable|numpy.ndarray) - 指定对角线值的输入张量，其秩应为1。

返回：存储着方阵的张量变量

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

        #  [3, 0, 0]
        #  [0, 4, 0]
        #  [0, 0, 5]
        data = fluid.layers.diag(np.arange(3, 6))




.. _cn_api_fluid_layers_fill_constant:

fill_constant
-------------------------------

.. py:function:: paddle.fluid.layers.fill_constant(shape,dtype,value,force_cpu=False,out=None)

该功能创建一个张量，含有具体的shape,dtype和batch尺寸。并用 ``value`` 中提供的常量初始化该张量。

创建张量的属性stop_gradient设为True。

参数：
    - **shape** (tuple|list|None)-输出张量的形状
    - **dtype** (np.dtype|core.VarDesc.VarType|str)-输出张量的数据类型
    - **value** (float)-用于初始化输出张量的常量值
    - **out** (Variable)-输出张量
    - **force_cpu** (True|False)-若设为true,数据必须在CPU上

返回：存储着输出的张量

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    data = fluid.layers.fill_constant(shape=[1], value=0, dtype='int64')









.. _cn_api_fluid_layers_fill_constant_batch_size_like:

fill_constant_batch_size_like
-------------------------------

.. py:function:: paddle.fluid.layers.fill_constant_batch_size_like(input,shape,dtype,value,input_dim_idx=0,output_dim_idx=0)

该功能创建一个张量，含有具体的shape,dtype和batch尺寸。并用 ``Value`` 中提供的常量初始化该张量。该批尺寸从输入张量中获取。它还将stop_gradient设置为True.

参数：
    - **input** (Variable)-张量，其第input_dim_idx维可指定batch_size
    - **shape** (INTS)-输出的形状
    - **dtype** (INT)-可以为numpy.dtype。输出数据类型。默认为float32
    - **value** (FLOAT)-默认为0.将要被填充的值
    - **input_dim_idx** (INT)-默认为0.输入批尺寸维的索引
    - **output_dim_idx** (INT)-默认为0.输出批尺寸维的索引

返回：具有特定形状和值的张量

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    like = fluid.layers.data(name='like', shape=[1], dtype='float32')
    data = fluid.layers.fill_constant_batch_size_like(
                input=like, shape=[1], value=0, dtype='int64')










.. _cn_api_fluid_layers_has_inf:

has_inf
-------------------------------

.. py:function:: paddle.fluid.layers.has_inf(x)

测试x是否包括一个无穷数

参数：
  - **x(variable)** - 用于被检查的Tensor/LoDTensor

返回： tensor变量存储输出值，包含一个bool型数值

返回类型：Variable

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    data = fluid.layers.data(name="input", shape=[4, 32, 32], dtype="float32")
    res = fluid.layers.has_inf(data)









.. _cn_api_fluid_layers_has_nan:

has_nan
-------------------------------

.. py:function:: paddle.fluid.layers.has_nan(x)

测试x是否包含NAN

参数：
  - **x(variable)** - 用于被检查的Tensor/LoDTensor

返回： tensor变量存储输出值，包含一个bool型数值

返回类型：Variable

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    data = fluid.layers.data(name="input", shape=[4, 32, 32], dtype="float32")
    res = fluid.layers.has_nan(data)




.. _cn_api_fluid_layers_isfinite:

isfinite
-------------------------------

.. py:function:: paddle.fluid.layers.isfinite(x)

测试x是否包含无穷大/NAN值，如果所有元素都是有穷数，返回Ture,否则返回False

参数：
  - **x(variable)** - 用于被检查的Tensor/LoDTensor

返回: Variable: tensor变量存储输出值，包含一个bool型数值

返回类型：Variable

**代码示例**：

.. code-block:: python

    var = fluid.layers.data(name="data",
                            shape=(4, 6),
                            dtype="float32")
    out = fluid.layers.isfinite(v)



.. _cn_api_fluid_layers_linspace:

linspace
-------------------------------

.. py:function:: paddle.fluid.layers.linspace(start, stop, num, dtype)

在给定区间内返回固定数目的均匀间隔的值。
 
第一个entry是start，最后一个entry是stop。在Num为1的情况下，仅返回start。类似numpy的linspace功能。

参数：
    - **start** (float|Variable)-序列中的第一个entry。 它是一个浮点标量，或是一个数据类型为'float32'|'float64'、形状为[1]的张量。
    - **stop** (float|Variable)-序列中的最后一个entry。 它是一个浮点标量，或是一个数据类型为'float32'|'float64'、形状为[1]的张量。
    - **num** (int|Variable)-序列中的entry数。 它是一个整型标量，或是一个数据类型为int32、形状为[1]的张量。
    - **dtype** (string)-‘float32’|’float64’，输出张量的数据类型。

返回：存储一维张量的张量变量

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

      data = fluid.layers.linspace(0, 10, 5, 'float32') # [0.0,  2.5,  5.0,  7.5, 10.0]
      data = fluid.layers.linspace(0, 10, 1, 'float32') # [0.0]





.. _cn_api_fluid_layers_ones:

ones
-------------------------------

.. py:function:: paddle.fluid.layers.ones(shape,dtype,force_cpu=False)

**ones**

该功能创建一个张量，有具体的维度和dtype，初始值为1。

也将stop_gradient设置为True。

参数：
    - **shape** (tuple|list)-输出张量的维
    - **dtype** (np.dtype|core.VarDesc.VarType|str)-输出张量的数据类型

返回：存储在输出中的张量

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    data = fluid.layers.ones(shape=[1], dtype='int64')



.. _cn_api_fluid_layers_range:

range
-------------------------------

.. py:function:: paddle.fluid.layers.range(start, end, step, dtype)

均匀分隔给定数值区间，并返回该分隔结果。

返回值在半开区间[start，stop)内生成（即包括起点start但不包括终点stop的区间）。


参数：
    - **start** （int | float | Variable） - 区间起点，且区间包括此值。
    - **end** （int | float | Variable） - 区间终点，通常区间不包括此值。但当step不是整数，且浮点数取整会影响out的长度时例外。
    - **step** （int | float | Variable） - 返回结果中数值之间的间距（步长）。 对于任何输出变量out，step是两个相邻值之间的距离，即out [i + 1]  -  out [i]。 默认为1。
    - **dtype** （string） - 'float32'|'int32'| ...，输出张量的数据类型。

返回：均匀分割给定数值区间后得到的值组


**代码示例**：

.. code-block:: python

    data = fluid.layers.range(0, 10, 2, 'int32')





.. _cn_api_fluid_layers_reverse:

reverse
-------------------------------

.. py:function:: paddle.fluid.layers.reverse(x,axis)

**reverse**

该功能将给定轴上的输入‘x’逆序

参数：
  - **x** (Variable)-预逆序的输入
  - **axis** (int|tuple|list) - 元素逆序排列的轴。如果该参数是一个元组或列表，则对该参数中每个元素值所指定的轴上进行逆序运算。

返回：逆序的张量

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

        import paddle.fluid as fluid
        data = fluid.layers.data(name="data", shape=[4, 8], dtype="float32")
        out = fluid.layers.reverse(x=data, axis=0)
        # or:
        out = fluid.layers.reverse(x=data, axis=[0,1])









.. _cn_api_fluid_layers_sums:

sums
-------------------------------

.. py:function:: paddle.fluid.layers.sums(input,out=None)

该函数对输入进行求和，并返回求和结果作为输出。

参数：
    - **input** (Variable|list)-输入张量，有需要求和的元素
    - **out** (Variable|None)-输出参数。求和结果。默认：None

返回：输入的求和。和参数'out'等同

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
     
    # sum of several tensors
    a0 = fluid.layers.fill_constant(shape=[1], dtype='int64', value=1)
    a1 = fluid.layers.fill_constant(shape=[1], dtype='int64', value=2)
    a2 = fluid.layers.fill_constant(shape=[1], dtype='int64', value=3)
    sums = fluid.layers.sums(input=[a0, a1, a2])

    # sum of a tensor array
    array = fluid.layers.create_array('int64')
    i = fluid.layers.zeros(shape=[1], dtype='int64', force_cpu=True)
    fluid.layers.array_write(a0, array=array, i=i)
    i = fluid.layers.increment(x=i)
    fluid.layers.array_write(a1, array=array, i=i)
    i = fluid.layers.increment(x=i)
    fluid.layers.array_write(a2, array=array, i=i)
    sums = fluid.layers.sums(input=array)









.. _cn_api_fluid_layers_tensor_array_to_tensor:

tensor_array_to_tensor
-------------------------------

.. py:function:: paddle.fluid.layers.tensor_array_to_tensor(input, axis=1, name=None)

此函数在指定轴上连接LodTensorArray中的元素，并将其作为输出返回。


简单示例如下：

.. code-block:: text

    Given:
    input.data = {[[0.6, 0.1, 0.3],
                   [0.5, 0.3, 0.2]],
                  [[1.3],
                   [1.8]],
                  [[2.3, 2.1],
                   [2.5, 2.4]]}

    axis = 1

    Then:
    output.data = [[0.6, 0.1, 0.3, 1.3, 2.3, 2.1],
                   [0.5, 0.3, 0.2, 1.8, 2.5, 2.4]]
    output_index.data = [3, 1, 2]

参数：
  - **input** (list) - 输入的LodTensorArray
  - **axis** (int) - 整数轴，tensor将会和它连接在一起
  - **name** (str|None) - 该layer的名字，可选。如果设置为none，layer将会被自动命名

返回：
    Variable: 连接的输出变量,输入LodTensorArray沿指定axis连接。

返回类型： Variable

**代码示例：**

.. code-block:: python

   import paddle.fluid as fluid
   tensor_array = fluid.layers.create_parameter(shape=[784, 200], dtype='float32')
   output, output_index = fluid.layers.tensor_array_to_tensor(input=tensor_array)











.. _cn_api_fluid_layers_zeros:

zeros
-------------------------------

.. py:function:: paddle.fluid.layers.zeros(shape,dtype,force_cpu=False)

**zeros**

该函数创建一个张量，含有具体的维度和dtype，初始值为0.

也将stop_gradient设置为True。

参数：
    - **shape** (tuple|list|None)-输出张量的维
    - **dtype** (np.dtype|core.VarDesc.VarType|str)-输出张量的数据类型
    - **force_cpu** (bool,default False)-是否将输出保留在CPU上

返回：存储在输出中的张量

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    data = fluid.layers.zeros(shape=[1], dtype='int64')








.. _cn_api_fluid_layers_zeros_like:

zeros_like
-------------------------------

.. py:function:: paddle.fluid.layers.zeros_like(x, out=None)

**zeros_like**

该函数创建一个和x具有相同的形状和数据类型的全零张量

参数：
    - **x** (Variable)-指定形状和数据类型的输入张量
    - **out** (Variable)-输出张量
    
返回：存储输出的张量变量

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    x = fluid.layers.data(name='x', dtype='float32', shape=[3], append_batch_size=False)
    data = fluid.layers.zeros_like(x) # [0.0, 0.0, 0.0]






