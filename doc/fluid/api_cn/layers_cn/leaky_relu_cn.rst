.. _cn_api_fluid_layers_leaky_relu:

leaky_relu
-------------------------------

.. py:function:: paddle.fluid.layers.leaky_relu(x, alpha=0.02, name=None)

LeakyRelu激活函数

.. math::   out=max(x,α∗x)

参数:
    - **x** (Variable) - 输入的多维 ``Tensor`` ，数据类型为：float32，float64。
    - **alpha** (float) - 负斜率，缺省值为0.02。
    - **name** (str，可选) - 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` ，默认值为None。

返回: 与 ``x`` 维度相同，数据类型相同的Tensor。

返回类型: Variable

**代码示例：**

.. code-block:: python

    import paddle.fluid as fluid
    import numpy as np

    # Graph Organizing
    x = fluid.layers.data(name="x", shape=[2], dtype="float32")
    res = fluid.layers.leaky_relu(x, alpha=0.1)

    # Create an executor using CPU as an example
    exe = fluid.Executor(fluid.CPUPlace())
    exe.run(fluid.default_startup_program())

    # Execute
    x_i = np.array([[-1, 2], [3, -4]]).astype(np.float32)
    res_val, = exe.run(fluid.default_main_program(), feed={'x':x_i}, fetch_list=[res])
    print(res_val) # [[-0.1, 2], [3, -0.4]]


