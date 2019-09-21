.. _cn_api_fluid_layers_sqrt:

sqrt
-------------------------------

.. py:function:: paddle.fluid.layers.sqrt(x, name=None)

计算输入的算数平方根。

.. math:: out=\sqrt x=x^{1/2}

<font color="#FF0000">**注意：请确保输入中的数值是非负数。**</font>

参数:

    - **x** (Variable) - 支持任意维度的Tensor。数据类型为float32，float64或float16。
    - **name** (str，可选) – 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` , 默认值为None。

返回：返回类型为Variable(Tensor|LoDTensor)， 数据类型同输入一致。

**代码示例**：

.. code-block:: python

        import numpy as np
        import paddle.fluid as fluid

        inputs = fluid.layers.data(name="x", shape = [3], dtype='float32')
        output = fluid.layers.sqrt(inputs)

        exe = fluid.Executor(fluid.CPUPlace())
        exe.run(fluid.default_startup_program())

        img = np.array([0, 9, 36]).astype(np.float32)

        res = exe.run(fluid.default_main_program(), feed={'x':img}, fetch_list=[output])
        print(res)
        # [array([0., 3., 6.], dtype=float32)] 













