.. _cn_api_fluid_layers_soft_relu:

soft_relu
-------------------------------

.. py:function:: paddle.fluid.layers.soft_relu(x, threshold=40.0, name=None)

SoftReLU 激活函数.

.. math::   out=ln(1+exp(max(min(x,threshold),threshold)))

参数:
    - **x** (Variable) - SoftReLU激活函数的输入，为数据类型为float32，float64的多维Tensor或者LoDTensor。
    - **threshold** (float) - SoftRelu的阈值，默认为40.0。
    - **name** (str，可选) - 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` ，默认值为None。

返回：一个Tensor，shape和输入Tensor相同。

返回类型：Variable(Tensor|LoDTensor)，LoD信息与输入Tensor一致。

**代码示例：**

.. code-block:: python

    import paddle.fluid as fluid
    import numpy as np

    inputs = fluid.layers.data(name="x", shape=[2, 2], dtype="float32")
    output = fluid.layers.soft_relu(inputs, threshold=20.0)

    exe = fluid.Executor(fluid.CPUPlace())
    exe.run(fluid.default_startup_program())

    img = np.array([[0, 1],[2, 3]]).astype(np.float32)

    res = exe.run(fluid.default_main_program(), feed={'x':img}, fetch_list=[output])
    print(res) # [array([[0.6931472, 1.3132616], [2.126928 , 3.0485873]], dtype=float32)]








