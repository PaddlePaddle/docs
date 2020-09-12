.. _cn_api_fluid_dygraph_NoamDecay:

NoamDecay
-------------------------------


.. py:class:: paddle.fluid.dygraph.NoamDecay(d_model, warmup_steps, begin=1, step=1, dtype='float32', learning_rate=1.0)




该接口提供Noam衰减学习率的功能。

Noam衰减的计算方式如下。

.. math::

    decayed\_learning\_rate = learning\_rate * d_{model}^{-0.5} * min(global\_steps^{-0.5}, global\_steps * warmup\_steps^{-1.5})

关于Noam衰减的更多细节请参考 `attention is all you need <https://arxiv.org/pdf/1706.03762.pdf>`_

式中，

- :math:`decayed\_learning\_rate` ： 衰减后的学习率。
式子中各参数详细介绍请看参数说明。

参数：
    - **d$_{model}$**  (Variable|int) - 模型的输入、输出向量特征维度，为超参数。如果设置为Variable类型值，则数据类型可以为int32或int64的标量Tensor，也可以设置为Python int。
    - **warmup_steps** (Variable|int) - 预热步数，为超参数。如果设置为Variable类型，则数据类型为int32或int64的标量Tensor，也可以设置为为Python int。
    - **begin** (int，可选) – 起始步。即以上运算式子中global_steps的初始值。默认值为0。
    - **step** (int，可选) – 步大小。即以上运算式子中global_steps的递增值。默认值为1。
    - **dtype** (str，可选) – 学习率值的数据类型，可以为"float32", "float64"。默认值为"float32"。
    - **learning_rate** (Variable|float|int，可选) - 初始学习率。如果类型为Variable，则为shape为[1]的Tensor，数据类型为float32或float64；也可以是python的int类型。默认值为1.0。

返回： 无

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    warmup_steps = 100
    learning_rate = 0.01
    with fluid.dygraph.guard():
        emb = fluid.dygraph.Embedding([10, 10])
        optimizer  = fluid.optimizer.SGD(
            learning_rate = fluid.dygraph.NoamDecay(
                   1/(warmup_steps *(learning_rate ** 2)),
                   warmup_steps),
            parameter_list = emb.parameters())
