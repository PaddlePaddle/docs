.. _cn_api_fluid_layers_equal:

equal
>>>>>>>>>>
.. py:class:: paddle.fluid.layers. equal(x,y,cond=None,**ignored)

**equal**
该层返回 :math:'x==y' 按逐元素运算而得的真值。

参数：
    - **x** (Variable)-equal的第一个操作数
    - **y** (Variable)-equal的第二个操作数
    - **cond** (Variable|None)-输出变量（可选），用来存储equal的结果

返回：张量类型的变量，存储equal的输出结果 

返回类型：变量（Variable） 

**代码示例**: 

.. code-block:: python

    less = fluid.layers.equal(x=label,y=limit)

.. _cn_api_fluid_layers_array_read:

array_read
>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers. array_read(array,i)

此函数用于读取数据，数据以LOD_TENSOR_ARRAY数组的形式读入

::
    Given:
    array = [0.6,0.1,0.3,0.1]
    And:
    I=2
    Then:
    output = 0.3

参数：
    - **array** (Variable|list)-输入张量，存储要读的数据
    - **i** (Variable|list)-输入数组中数据的索引

返回：张量类型的变量，已有数据写入

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    tmp = fluid.layers.zeros(shape=[10],dtype='int32')
    i = fluid.layers.fill_constant(shape=[1],dtype='int64',value=10)
    arr = layers.array_read(tmp,i=i)

.. _cn_api_fluid_layers_array_length:

array_length
>>>>>>>>>>>>>>
.. py:class:: paddle.fluid.layers. array_length(array)

**得到输入LoDTensorArray的长度**

此功能用于查找输入数组LOD_TENSOR_ARRAY的长度。  

相关API:
    - ref:'api_fluid_layers_array_read',
    - ref:'api_fluid_layers_array_write',
    - ref:'api_fluid_layers_While'. 

参数：**array** (LOD_TENSOR_ARRAY)-输入数组，用来计算数组长度

返回：输入数组LoDTensorArray的长度

返回类型：变量（Variable）

**代码示例**:

.. code-block:: python

    tmp = fluid.layers.zeros(shape=[10], dtype='int32')
    i = fluid.layers.fill_constant(shape=[1], dtype='int64', value=10)
    arr = fluid.layers.array_write(tmp, i=i)
    arr_len = fluid.layers.array_length(arr)

.. _cn_api_fluid_layers_IfElse:

IfElse
>>>>>>>

.. py:class:: paddle.fluid.layers. IfElse(cond, name=None)

if-else控制流。  

参数：
    - **cond** (Variable)-用于比较的条件
    - **Name** (str,默认为空（None）)-该层名称

**代码示例**：

.. code-block:: python

    limit = fluid.layers.fill_constant_batch_size_like(
        input=label, dtype='int64', shape=[1], value=5.0)
    cond = fluid.layers.less_than(x=label, y=limit)
    ie = fluid.layers.IfElse(cond)
    with ie.true_block():
        true_image = ie.input(image)
        hidden = fluid.layers.fc(input=true_image, size=100, act='tanh')
        prob = fluid.layers.fc(input=hidden, size=10, act='softmax')
        ie.output(prob)

    with ie.false_block():
        false_image = ie.input(image)
        hidden = fluid.layers.fc(
            input=false_image, size=200, act='tanh')
        prob = fluid.layers.fc(input=hidden, size=10, act='softmax')
        ie.output(prob)
    prob = ie()

.. _cn_api_fluid_layers_Print:

Print
>>>>>>>
.. py:class:: paddle.fluid.layers. Print(input, first_n=-1, message=None, summarize=-1, print_tensor_name=True, print_tensor_type=True, print_tensor_shape=True, print_tensor_lod=True, print_phase='both')

**Print操作命令**

该操作命令创建一个打印操作，打印正在访问的张量。

封装传入的张量，以便无论何时访问张量，都会打印信息message和张量的当前值。

参数：

    - **input** (Variable)-将要打印的张量
    - **summarize** (int)-打印张量中的元素数目，如果值为-1则打印所有元素
    - **message** (str)-字符串类型消息，作为前缀打印
    - **first_n** (int)-只记录first_n次数
    - **print_tensor_name** (bool)-打印张量名称
    - **print_tensor_type** (bool)-打印张量类型
    - **print_tensor_shape** (bool)-打印张量维度
    - **print_tensor_lod** (bool)-打印张量lod
    - **print_phase** (str)-打印的阶段，包括"forward","backward"和"both".若设置为"backward"或者"both",则打印输入张量的梯度。

返回：输出张量，和输入张量同样的数据

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    value = some_layer(...)
    Print(value, summarize=10,
    message="The content of some_layer: ")

.. _cn_api_fluid_layers_is_empty:

is_empty
>>>>>>>>>
.. py:class:: paddle.fluid.layers. is_empty(x, cond=None, **ignored)

测试变量是否为空

参数：
    - **x** (Variable)-测试的变量
    - **cond** (Variable|None)-输出参数。返回给定x的测试结果，默认为空（None）

返回：布尔类型的标量。如果变量x为空则值为真

返回类型：变量（Variable）

抛出异常：''TypeError''-如果input不是变量或cond类型不是变量

**代码示例**：

.. code-block:: python

    res = fluid.layers.is_empty(x=input)
    # or:
    fluid.layers.is_empty(x=input, cond=res)