.. _cn_api_fluid_dygraph_NoamDecay:

NoamDecay
-------------------------------

.. py:class:: paddle.fluid.dygraph.NoamDecay(d_model, warmup_steps, begin=1, step=1, dtype='float32')

Noam衰减方法。noam衰减的numpy实现如下。

.. code-block:: python

    import numpy as np
    # 设置超参数
    d_model = 2
    current_steps = 20
    warmup_steps = 200
    # 计算
    lr_value = np.power(d_model, -0.5) * np.min([
                           np.power(current_steps, -0.5),
                           np.power(warmup_steps, -1.5) * current_steps])

请参照 `attention is all you need <https://arxiv.org/pdf/1706.03762.pdf>`_

参数：
    - **d_model** (Variable)-模型的输入和输出维度
    - **warmup_steps** (Variable)-超参数
    - **begin**  – 起始步(默认为0)。
    - **step**  – 步大小(默认为1)。
    - **dtype**  – 初始学习率的dtype(默认为‘float32’)。

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



