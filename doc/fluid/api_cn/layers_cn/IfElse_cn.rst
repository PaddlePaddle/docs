.. _cn_api_fluid_layers_IfElse:

IfElse
-------------------------------

.. py:class:: paddle.fluid.layers.IfElse(cond, name=None)

该类用于实现IfElse分支控制功能， IfElse包含两个Block，true_block, false_block，IfElse会将满足True或False条件的数据分别放入不同的block运行。    

cond是一个shape为[N, 1]、数据类型为bool的2-D tensor，表示输入数据对应部分的执行条件。

IfElse OP同其他的OP在使用上有一定的区别，可能会对一些用户造成一定的困惑，以下展示了一个
简单的样例对该OP进行说明。

.. code-block:: python
        
        # 以下代码完成的功能：对x中大于0的数据减去10，对x中小于0的数据加上10
        import numpy as np
        import paddle.fluid as fluid

        x = fluid.layers.data(name='x', shape=[4, 1], dtype='float32', append_batch_size=False)
        y = fluid.layers.data(name='y', shape=[4, 1], dtype='float32', append_batch_size=False)

        x_d = np.array([[3], [1], [-1], [-3]]).astype(np.float32)
        y_d = np.zeros((4, 1)).astype(np.float32)
        
        # 比较x, y对元素的大小，输出cond, cond是shape为[4, 1]，数据类型为bool的2-D tensor。
        # 根据输入数据x_d, y_d，可以推断出cond中的数据为[[true], [true], [false], [false]]
        cond = fluid.layers.greater_than(x, y)
        # 同其他常见OP不同的是，该OP返回的ie不是Variable而是一个IfElse OP的对象
        ie = fluid.layers.IfElse(cond)

        with ie.true_block():
            # 在这个block中，根据cond条件，获取x中对应条件为true维度的数据，并减去10
            out_1 = ie.input(x)
            out_1 = out_1 - 10
            ie.output(out_1)
        with ie.false_block():
            # 在这个block中，根据cond条件，获取x中对应条件为false维度的数据，并加上10
            out_1 = ie.input(x)
            out_1 = out_1 + 10
            ie.output(out_1)

        # 根据cond条件将两个block中处理后的数据进行合并，此处的output为输出的Variable
        output = ie()

        exe = fluid.Executor(fluid.CPUPlace())
        exe.run(fluid.default_startup_program())

        res = exe.run(fluid.default_main_program(), feed={"x":x_d, "y":y_d}, fetch_list=[output[0]])
        print res
        # [array([[-7.],
        #         [-9.],
        #         [ 9.],
        #         [ 7.]], dtype=float32)] 

参数：
    - **cond** (Variable)- cond是一个shape为[N, 1]、数据类型为bool的2-D tensor，表示N个输入数据的对应的执行条件。数据类型为bool。
    - **Name** (str，可选)- 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` , 默认值为None。

**返回：**    

同其他常见OP不同的是：该OP调用返回一个IfElse OP对象(如例子中的 ie)而非Variable，需要调用对象内部函数对不同cond下的输入进行处理。

**内部函数：**

通过调用对象中的 ``with ie.true_block()`` 函数构建block，将条件为true下的计算逻辑放入此block中。如果没有构建相应的block，则对应条件维度下的输入数据不做改变。

``out = ie.input(x)`` 会将x中对应条件维度的数据获取出来放入到out中，支持block内部处理多个输入。

``ie.output(out)`` 会将结果写入对应条件的输出中。

对象内部有 ``__call__()`` 函数，即通过对 ``output = ie()`` 的调用，将条件分别为True，False的block内部所有的输出进行融合作为整体的输出，输出的类型为列表，列表中每个元素的类型为Variable。




