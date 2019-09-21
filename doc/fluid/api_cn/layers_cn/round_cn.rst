.. _cn_api_fluid_layers_round:

round
-------------------------------

.. py:function:: paddle.fluid.layers.round(x, name=None)


该OP将输入中的数值四舍五入到最接近的整数数值。

.. code-block:: python

  输入：
    x.shape = [4]
    x.data = [1.2, -0.9, 3.4, 0.9]

  输出：
    out.shape = [4]
    Out.data = [1., -1., 3., 1.]

参数:

    - **x** (Variable) - 支持任意维度的Tensor。数据类型为float32，float64或float16。
    - **name** (str，可选) – 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` , 默认值为None。

返回：返回类型为Variable(Tensor|LoDTensor)， 数据类型同输入一致。

**代码示例**：

.. code-block:: python

        import numpy as np
        import paddle.fluid as fluid

        inputs = fluid.layers.data(name="x", shape = [3], dtype='float32')
        output = fluid.layers.round(inputs)

        exe = fluid.Executor(fluid.CPUPlace())
        exe.run(fluid.default_startup_program())

        img = np.array([1.2, -0.9, 3.4, 0.9]).astype(np.float32)

        res = exe.run(fluid.default_main_program(), feed={'x':img}, fetch_list=[output])
        print(res)
        # [array([ 1., -1.,  3.,  1.], dtype=float32)]



