.. _cn_api_fluid_layers_Print:

Print
-------------------------------

.. py:function:: paddle.fluid.layers.Print(input, first_n=-1, message=None, summarize=-1, print_tensor_name=True, print_tensor_type=True, print_tensor_shape=True, print_tensor_lod=True, print_phase='both')

**Print操作命令**

该OP创建一个打印操作，打印正在访问的Tensor。

封装传入的Tensor，以便无论何时访问Tensor，都会打印信息message和Tensor的当前值。

参数：
    - **input** (Variable)-将要打印的Tensor
    - **summarize** (int)-打印Tensor中的元素数目，如果值为-1则打印所有元素
    - **message** (str)-打印Tensor信息前自定义的字符串类型消息，作为前缀打印
    - **first_n** (int)-打印Tensor的次数
    - **print_tensor_name** (bool)-可选，指明是否打印Tensor名称，默认为True
    - **print_tensor_type** (bool)-可选，指明是否打印Tensor类型，默认为True
    - **print_tensor_shape** (bool)-可选，指明是否打印Tensor维度信息，默认为True
    - **print_tensor_lod** (bool)-可选，指明是否打印Tensor的lod信息，默认为True
    - **print_phase** (str)-可选，指明打印的阶段，包括 ``forward`` , ``backward`` 和 ``both`` 。默认为 ``both`` 。设置为 ``forward`` 时，只打印Tensor的前向信息；设置为 ``backward`` 时，只打印Tensor的梯度信息；设置为 ``both`` 时，则同时打印Tensor的前向信息以及梯度信息。

返回：输出Tensor

返回类型：Variable

.. note::
   输入和输出是两个不同的变量，在接下来的过程中，你应该使用输出变量而非输入变量，否则打印层将失去backward的信息。

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    import paddle
    import numpy as np

    x = fluid.layers.data(name='x', shape=[1], dtype='float32', lod_level=1)
    x = fluid.layers.Print(x, message="The content of input layer:")
    
    y = fluid.layers.data(name='y', shape=[1], dtype='float32', lod_level=2)
    out = fluid.layers.sequence_expand(x=x, y=y, ref_level=0)
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    x_d = fluid.create_lod_tensor(np.array([[1.1], [2.2],[3.3],[4.4]]).astype('float32'), [[1,3]], place)
    y_d = fluid.create_lod_tensor(np.array([[1.1],[1.1],[1.1],[1.1],[1.1],[1.1]]).astype('float32'), [[1,3], [1,2,1,2]], place)
    results = exe.run(fluid.default_main_program(),
                      feed={'x':x_d, 'y': y_d },
                      fetch_list=[out],return_numpy=False)
**运行输出**:

.. code-block:: bash 
   
   The content of input layer:    The place is:CPUPlace
   Tensor[x]
    shape: [4,1,]
    dtype: f
    LoD: [[ 0,1,4, ]]
    data: 1.1,2.2,3.3,4.4,





