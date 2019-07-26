.. _cn_api_fluid_layers_Print:

Print
-------------------------------

.. py:function:: paddle.fluid.layers.Print(input, first_n=-1, message=None, summarize=-1, print_tensor_name=True, print_tensor_type=True, print_tensor_shape=True, print_tensor_lod=True, print_phase='both')

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
    - **print_phase** (str)-打印的阶段，包括 ``forward`` , ``backward`` 和 ``both`` .若设置为 ``backward`` 或者 ``both`` ,则打印输入张量的梯度。

返回：输出张量

返回类型：变量（Variable）

.. note::
   输入和输出是两个不同的变量，在接下来的过程中，你应该使用输出变量而非输入变量，否则打印层将失去输出层前的信息。

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
     
    input = fluid.layers.data(name="input", shape=[4, 32, 32], dtype="float32")
    input = fluid.layers.Print(input, message = "The content of input layer:")
    # value = some_layer(...)
    # Print(value, summarize=10,
    #     message="The content of some_layer: ")









