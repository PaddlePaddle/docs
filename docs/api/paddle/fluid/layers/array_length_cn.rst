.. _cn_api_fluid_layers_array_length:

array_length
-------------------------------

.. py:function:: paddle.fluid.layers.array_length(array)




该 OP 用于获取输入数组 :ref:`cn_api_fluid_LoDTensorArray` 的长度。可以与 :ref:`cn_api_fluid_layers_array_read` 、 :ref:`cn_api_fluid_layers_array_write` 、 :ref:`cn_api_fluid_layers_While` OP 结合使用，实现 LoDTensorArray 的遍历与读写。

参数
::::::::::::

    - **array** (LoDTensorArray) - 输入的数组 LoDTensorArray

返回
::::::::::::
shape 为[1]的 1-D Tensor，表示数组 LoDTensorArray 的长度，数据类型为 int64

返回类型
::::::::::::
Variable

代码示例
::::::::::::

.. code-block:: python

    import paddle.fluid as fluid
    tmp = fluid.layers.zeros(shape=[10], dtype='int32')
    i = fluid.layers.fill_constant(shape=[1], dtype='int64', value=10)
    #tmp 是 shape 为[10]的 Tensor，将 tmp 写入到数组 arr 下标为 10 的位置，arr 的长度因此为 11
    arr = fluid.layers.array_write(tmp, i=i)
    #查看 arr 的长度
    arr_len = fluid.layers.array_length(arr)

    #可以通过 executor 打印出 LoDTensorArray 的长度
    input = fluid.layers.Print(arr_len, message="The length of LoDTensorArray:")
    main_program = fluid.default_main_program()
    exe = fluid.Executor(fluid.CPUPlace())
    exe.run(main_program)

**运行输出**

COPY-FROM: paddle.fluid.layers.array_length
