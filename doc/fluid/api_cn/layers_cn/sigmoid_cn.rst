.. _cn_api_fluid_layers_sigmoid:

sigmoid
-------------------------------

.. py:function:: paddle.fluid.layers.sigmoid(x, name=None)

sigmoid激活函数

.. math::
    out = \frac{1}{1 + e^{-x}}


参数：

    - **x** - Sigmoid算子的输入
    - **name** (str|None) - 该层名称（可选）。若为空，则自动为该层命名。默认：None

返回： 张量，sigmoid计算结果

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

        import paddle.fluid as fluid
        import numpy as np
        data = fluid.layers.data(name="input", shape=[-1, 3])
        result = fluid.layers.sigmoid(data)
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
        x = np.random.random(size=(3, 3)).astype('float32')
        output= exe.run(feed={"input": x},
                         fetch_list=[result])
        print(output)











