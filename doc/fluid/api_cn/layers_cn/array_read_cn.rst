.. _cn_api_fluid_layers_array_read:

array_read
-------------------------------

.. py:function:: paddle.fluid.layers.array_read(array,i)




该OP用于读取输入数组 :ref:`cn_api_fluid_LoDTensorArray` 中指定位置的数据, ``array`` 为输入的数组， ``i`` 为指定的读取位置。常与 :ref:`cn_api_fluid_layers_array_write` OP配合使用进行LoDTensorArray的读写。

例1:
::
    输入：
        包含4个Tensor的LoDTensorArray，前3个shape为[1]，最后一个shape为[1,2]:
            input = ([0.6], [0.1], [0.3], [0.4, 0.2])
        并且:
            i = [3]

    输出:
        output = [0.4, 0.2]

参数：
    - **array** (Variable) - 输入的数组LoDTensorArray
    - **i** (Variable) - shape为[1]的1-D Tensor，表示从 ``array`` 中读取数据的位置，数据类型为int64


返回：从 ``array`` 中指定位置读取的LoDTensor或Tensor

返回类型：Variable

**代码示例**

.. code-block:: python

    #先创建一个LoDTensorArray，再在指定位置写入Tensor，然后从该位置读取Tensor
    import paddle
    import paddle.fluid as fluid
    arr = fluid.layers.create_array(dtype='float32')
    #读取arr的下标10的位置的数据
    tmp = paddle.full(shape=[3, 2], dtype='int64', fill_value=5, device=None,
        stop_gradient=True)
    i = paddle.full(shape=[1], dtype='int64', fill_value=10, device=None,
    #tmp是shape为[3,2]的Tensor，将其写入空数组arr的下标10的位置，则arr的长度变为11
        stop_gradient=True)
    arr = fluid.layers.array_write(tmp, i, array=arr)
    #读取arr的下标10的位置的数据
    item = fluid.layers.array_read(arr, i)
    
    #可以通过executor打印出该数据
    input = paddle.Print(item, message='The LoDTensor of the i-th position:')
    main_program = paddle.default_main_program()
    exe = paddle.Executor(paddle.CPUPlace())
    exe.run(main_program)

**输出结果**

.. code-block:: python

    #先创建一个LoDTensorArray，再在指定位置写入Tensor，然后从该位置读取Tensor
    import paddle
    import paddle.fluid as fluid
    arr = fluid.layers.create_array(dtype='float32')
    #读取arr的下标10的位置的数据
    tmp = paddle.full(shape=[3, 2], dtype='int64', fill_value=5, device=None,
        stop_gradient=True)
    i = paddle.full(shape=[1], dtype='int64', fill_value=10, device=None,
    #tmp是shape为[3,2]的Tensor，将其写入空数组arr的下标10的位置，则arr的长度变为11
        stop_gradient=True)
    arr = fluid.layers.array_write(tmp, i, array=arr)
    #读取arr的下标10的位置的数据
    item = fluid.layers.array_read(arr, i)
    
    #可以通过executor打印出该数据
    input = paddle.Print(item, message='The LoDTensor of the i-th position:')
    main_program = paddle.default_main_program()
    exe = paddle.Executor(paddle.CPUPlace())
    exe.run(main_program)

	    shape: [3,2,]
	    dtype: l
	    data: 5,5,5,5,5,5,

    #输出了shape为[3,2]的Tensor
    #dtype为对应C++数据类型，在不同环境下可能显示值不同，但本质一致
    #例如：如果Tensor中数据类型是int64，则对应的C++数据类型为int64_t，所以dtype值为typeid(int64_t).name()，
    #      其在MacOS下为'x'，linux下为'l'，Windows下为'__int64'，都表示64位整型变量
