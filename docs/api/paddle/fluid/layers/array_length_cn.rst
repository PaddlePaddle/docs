.. _cn_api_fluid_layers_array_length:

array_length
-------------------------------

.. py:function:: paddle.fluid.layers.array_length(array)




该OP用于获取输入数组 :ref:`cn_api_fluid_LoDTensorArray` 的长度。可以与 :ref:`cn_api_fluid_layers_array_read` 、 :ref:`cn_api_fluid_layers_array_write` 、 :ref:`cn_api_fluid_layers_While` OP结合使用，实现LoDTensorArray的遍历与读写。

参数：
    - **array** (LoDTensorArray) - 输入的数组LoDTensorArray

返回：shape为[1]的1-D Tensor, 表示数组LoDTensorArray的长度，数据类型为int64

返回类型：Variable

**代码示例**:

.. code-block:: python

    import paddle.fluid as fluid
    tmp = fluid.layers.zeros(shape=[10], dtype='int32')
    i = fluid.layers.fill_constant(shape=[1], dtype='int64', value=10)
    #tmp是shape为[10]的Tensor，将tmp写入到数组arr下标为10的位置，arr的长度因此为11
    arr = fluid.layers.array_write(tmp, i=i)
    #查看arr的长度
    arr_len = fluid.layers.array_length(arr)

    #可以通过executor打印出LoDTensorArray的长度
    input = fluid.layers.Print(arr_len, message="The length of LoDTensorArray:")
    main_program = fluid.default_main_program()
    exe = fluid.Executor(fluid.CPUPlace())
    exe.run(main_program)

**运行输出**

.. code-block:: python

    1569576542	The length of LoDTensorArray:	The place is:CPUPlace
    Tensor[array_length_0.tmp_0]
	shape: [1,]
	dtype: l
	data: 11,
    
    #输出shape为[1]的Tensor，值为11，表示LoDTensorArray长度为11
    #dtype为对应C++数据类型，在不同环境下可能显示值不同，但本质一致
    #例如：如果Tensor中数据类型是int64，则对应的C++数据类型为int64_t，所以dtype值为typeid(int64_t).name()，
    #      其在MacOS下为'x'，linux下为'l'，Windows下为'__int64'，都表示64位整型变量
