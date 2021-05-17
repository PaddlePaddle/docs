.. _cn_api_fluid_layers_Print:

Print
-------------------------------


.. py:function:: paddle.static.Print(input, first_n=-1, message=None, summarize=20, print_tensor_name=True, print_tensor_type=True, print_tensor_shape=True, print_tensor_lod=True, print_phase='both')




**Print操作命令**

该OP创建一个打印操作，打印正在访问的Tensor内容。

封装传入的Tensor，以便无论何时访问Tensor，都会打印信息message和Tensor的当前值。

参数：
    - **input** (Variable)-将要打印的Tensor
    - **summarize** (int)-打印Tensor中的元素数目，如果值为-1则打印所有元素。默认值为20
    - **message** (str)-打印Tensor信息前自定义的字符串类型消息，作为前缀打印
    - **first_n** (int)-打印Tensor的次数
    - **print_tensor_name** (bool)-可选，指明是否打印Tensor名称，默认为True
    - **print_tensor_type** (bool)-可选，指明是否打印Tensor类型，默认为True
    - **print_tensor_shape** (bool)-可选，指明是否打印Tensor维度信息，默认为True
    - **print_tensor_lod** (bool)-可选，指明是否打印Tensor的LoD信息，默认为True
    - **print_phase** (str)-可选，指明打印的阶段，包括 ``forward`` , ``backward`` 和 ``both`` ，默认为 ``both`` 。设置为 ``forward`` 时，只打印Tensor的前向信息；设置为 ``backward`` 时，只打印Tensor的梯度信息；设置为 ``both`` 时，则同时打印Tensor的前向信息以及梯度信息。

返回：输出Tensor

.. note::
   输入和输出是两个不同的Variable，在接下来的过程中，应该使用输出Variable而非输入Variable，否则打印层将失去backward的信息。

**代码示例**：

.. code-block:: python

    import paddle

    paddle.enable_static()

    x = paddle.full(shape=[2, 3], fill_value=3, dtype='int64')
    out = paddle.static.Print(x, message="The content of input layer:")

    main_program = paddle.static.default_main_program()
    exe = paddle.static.Executor(place=paddle.CPUPlace())
    res = exe.run(main_program, fetch_list=[out])
    # Variable: fill_constant_1.tmp_0
    #   - message: The content of input layer:
    #   - lod: {}
    #   - place: CPUPlace
    #   - shape: [2, 3]
    #   - layout: NCHW
    #   - dtype: long
    #   - data: [3 3 3 3 3 3]

