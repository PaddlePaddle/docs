.. _cn_api_fluid_layers_sigmoid:

sigmoid
-------------------------------

.. py:function:: paddle.fluid.layers.sigmoid(x, name=None)

sigmoid激活函数

.. math::
    out = \frac{1}{1 + e^{-x}}


参数：

    - **x** (Tensor|LoDTensor)- 数据类型为float32，float64。激活函数的输入值。
    - **name** (str|None) - 该层名称（可选）。若为空，则自动为该层命名。默认：None

返回：激活函数的输出值

返回类型：Variable（Tensor），数据类型为float32的Tensor。

**代码示例**：

.. code-block:: python

        import paddle.fluid as fluid
        import numpy as np

        data = fluid.layers.data(name="input", shape=[-1, 3])
        result = fluid.layers.sigmoid(data)
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
        x = np.random.rand(3, 3)
        output= exe.run(feed={"input": x},
                         fetch_list=[result[0]])
        print(output)
        """
        output:
        [array([0.50797188, 0.71353652, 0.5452265 ])]
        """










