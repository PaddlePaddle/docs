.. _cn_api_fluid_dygraph_NoamDecay:

NoamDecay
-------------------------------

.. py:class:: paddle.fluid.dygraph.NoamDecay(d_model, warmup_steps, begin=1, step=1, dtype='float32')

该接口提供一种Noam衰减方法。

以numpy实现的Noam衰减的计算方式如下。

.. code-block:: python

    import numpy as np
    # 设置超参数
    d_model = 2
    current_steps = 20
    warmup_steps = 200
    # 计算学习率
    lr_value = np.power(d_model, -0.5) * np.min([
                           np.power(current_steps, -0.5),
                           np.power(warmup_steps, -1.5) * current_steps])

关于Noam衰减的细节请参照 `attention is all you need <https://arxiv.org/pdf/1706.03762.pdf>`_

参数：
    - **d_model** (Variable|int) - 模型的输入、输出向量特征维度，为超参数。如果设置为Variable类型值，则数据类型可以为int32或int64的标量Tensor，也可以设置为Python int。
    - **warmup_steps** (Variable|int) - 预热步数，为超参数。如果设置为Variable类型，则数据类型为int32或int64的标量Tensor，也可以设置为为Python int。
    - **begin** (int，可选) – 起始步。即以上运算式子中current_steps的初始值。默认值为0。
    - **step** (int，可选) – 步大小。即以上运算式子中current_steps的递增值。默认值为1。
    - **dtype** (str，可选) – 学习率值的数据类型。默认值为‘float32’。

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    warmup_steps = 100
    learning_rate = 0.01
    with fluid.dygraph.guard():
        optimizer  = fluid.optimizer.SGD(
            learning_rate = fluid.dygraph.NoamDecay(
                   1/(warmup_steps *(learning_rate ** 2)),
                   warmup_steps) )



