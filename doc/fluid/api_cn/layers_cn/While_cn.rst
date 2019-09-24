.. _cn_api_fluid_layers_While:

While
-------------------------------

.. py:class:: paddle.fluid.layers.While (cond, is_test=False, name=None)


该类用于实现while循环控制功能，只要循环条件cond为True，就循环执行while循环体中的语句，直到cond为False为止。


参数：
    - **cond** (Variable) – 用于判断循环继续进行的条件，为数据类型为多维bool型的Tensor。
    - **is_test** (bool，可选) – 用于表明是否在测试阶段执行，默认值为False。
    - **name** (str，可选) - 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` ，默认值为None。

**代码示例**

..  code-block:: python

    # 该示例代码循环打印LOD_TENSOR_ARRAY中的Tensor
    import paddle.fluid as fluid
    import numpy as np

    i = fluid.layers.fill_constant(shape=[1], dtype='int64', value=0)
    d0 = fluid.layers.data("x0", shape=[3], dtype='float32')
    d1 = fluid.layers.data("x1", shape=[3], dtype='float32')

    data_array = fluid.layers.array_write(x=d0, i=i)
    i = fluid.layers.increment(x=i, value=1, in_place=True)
    data_array = fluid.layers.array_write(x=d1, i=i, array=data_array)

    j = fluid.layers.fill_constant(shape=[1], dtype='int64', value=0)        # 循环计数
    array_len = fluid.layers.fill_constant(shape=[1],dtype='int64', value=2) # 设定LOD_TENSOR_ARRAY长度，亦即循环次数

    cond = fluid.layers.less_than(x=j, y=array_len)         # 循环条件
    while_op = fluid.layers.While(cond=cond)
    with while_op.block(): # 循环体
        d = fluid.layers.array_read(array=data_array, i=j)
        fluid.layers.Print(d)
        j = fluid.layers.increment(x=j, value=1, in_place=True)
        
        fluid.layers.less_than(x=j, y=array_len, cond=cond) # 更新循环条件   

    exe = fluid.Executor(fluid.CPUPlace())
    exe.run(fluid.default_startup_program())

    img0 = np.array([1, 2, 3]).astype(np.float32)
    img1 = np.array([4, 5, 6]).astype(np.float32)

    exe.run(fluid.default_main_program(), feed={'x0':img0, 'x1':img1}, fetch_list=[])
    # 运行结果可能因具体运行环境不同而不同
    # 
    # 1569319544		The place is:CPUPlace
    # Tensor[array_read_0.tmp_0]
    # 	shape: [3,]
    # 	dtype: f
    # 	data: 1,2,3,
    # 1569319544		The place is:CPUPlace
    # Tensor[array_read_0.tmp_0]
    # 	shape: [3,]
    # 	dtype: f
    # 	data: 4,5,6,










