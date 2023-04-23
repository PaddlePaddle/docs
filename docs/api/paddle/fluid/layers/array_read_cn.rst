.. _cn_api_fluid_layers_array_read:

array_read
-------------------------------

.. py:function:: paddle.fluid.layers.array_read(array,i)




该OP用于读取输入数组 :ref:`cn_api_fluid_LoDTensorArray` 中指定位置的数据，``array`` 为输入的数组，``i`` 为指定的读取位置。常与 :ref:`cn_api_fluid_layers_array_write` OP配合使用进行LoDTensorArray的读写。

例1:
::
    输入：
        包含4个Tensor的LoDTensorArray，前3个shape为[1]，最后一个shape为[1,2]:
            input = ([0.6], [0.1], [0.3], [0.4, 0.2])
        并且：
            i = [3]

    输出：
        output = [0.4, 0.2]

参数
::::::::::::

    - **array** (Variable) - 输入的数组LoDTensorArray
    - **i** (Variable) - shape为[1]的1-D Tensor，表示从 ``array`` 中读取数据的位置，数据类型为int64


返回
::::::::::::
从 ``array`` 中指定位置读取的LoDTensor或Tensor

返回类型
::::::::::::
Variable

代码示例
::::::::::::

.. code-block:: python

    #先创建一个LoDTensorArray，再在指定位置写入Tensor，然后从该位置读取Tensor
    import paddle.fluid as fluid
    arr = fluid.layers.create_array(dtype='float32')
    tmp = fluid.layers.fill_constant(shape=[3, 2], dtype='int64', value=5)
    i = fluid.layers.fill_constant(shape=[1], dtype='int64', value=10)
    #tmp是shape为[3,2]的Tensor，将其写入空数组arr的下标10的位置，则arr的长度变为11
    arr = fluid.layers.array_write(tmp, i, array=arr)
    #读取arr的下标10的位置的数据
    item = fluid.layers.array_read(arr, i)

    #可以通过executor打印出该数据
    input = fluid.layers.Print(item, message="The LoDTensor of the i-th position:")
    main_program = fluid.default_main_program()
    exe = fluid.Executor(fluid.CPUPlace())
    exe.run(main_program)

**输出结果**

COPY-FROM: paddle.fluid.layers.array_read