.. _cn_api_fluid_layers_hard_shrink:

hard_shrink
-------------------------------

.. py:function:: paddle.fluid.layers.hard_shrink(x,threshold=None)

HardShrink激活函数(HardShrink activation operator)


.. math::

  out = \begin{cases}
        x, \text{if } x > \lambda \\
        x, \text{if } x < -\lambda \\
        0,  \text{otherwise}
      \end{cases}

参数：
    - **x**(Tensor|LoDTensor） - 数据类型为float64或者float32的Tensor或者LoDTensor。HardShrink激活函数的输入
    - **threshold**(FLOAT) - HardShrink激活函数的threshold值。[默认：0.5]

返回：HardShrink激活函数的输出
返回类型：Variable（Tensor），数据类型为float64或者float32的Tensor。

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    import numpy as np
    data = fluid.layers.data(name="input", shape=[-1, 2])
    result = fluid.layers.hard_shrink(data, threshold=0.4)
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    x = np.random.random(size=(3, 2)).astype('float64')
    output= exe.run(feed={"input": x},
                 fetch_list=[result])
    print(output)
    """
    output:
    [array([[0., 0.74170118],
            [0., 0.83063472],
            [0.89835296, 0.]])]
    """







