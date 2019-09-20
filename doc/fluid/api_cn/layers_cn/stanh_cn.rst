.. _cn_api_fluid_layers_stanh:

stanh
-------------------------------

.. py:function:: paddle.fluid.layers.stanh(x, scale_a=0.67, scale_b=1.7159, name=None)

STanh 激活算子（STanh Activation Operator.）

.. math::
          \\out=b*\frac{e^{a*x}-e^{-a*x}}{e^{a*x}+e^{-a*x}}\\

参数：
    - **x** (Variable) - STanh operator的输入
    - **scale_a** (FLOAT|2.0 / 3.0) - 输入的a的缩放参数
    - **scale_b** (FLOAT|1.7159) - b的缩放参数
    - **name** (str|None) - 这个层的名称(可选)。如果设置为None，该层将被自动命名。

返回: 张量，激活函数STanh操作符的输出

返回类型: 输出(Variable)

**代码示例：**

.. code-block:: python

    import paddle.fluid as fluid
    import numpy as np
    data = fluid.layers.data(name="input", shape=[-1, 3])
    result = fluid.layers.stanh(data,scale_a=0.67, scale_b=1.72)
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    x = np.random.random(size=(3, 3)).astype('float32')
    output= exe.run(feed={"input": x},
                 fetch_list=[result])
    print(output)




