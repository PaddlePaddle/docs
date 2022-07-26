.. _cn_api_fluid_layers_noam_decay:

noam_decay
-------------------------------

.. py:function:: paddle.fluid.layers.noam_decay(d_model, warmup_steps)




Noam衰减方法

noam衰减的numpy实现如下：

.. code-block:: python

    import paddle.fluid as fluid
    import numpy as np
    # 设置超参数
    base_lr = 0.01
    d_model = 2
    current_steps = 20
    warmup_steps = 200
    # 计算
    lr_value = base_lr * np.power(d_model, -0.5) * np.min([
                           np.power(current_steps, -0.5),
                           np.power(warmup_steps, -1.5) * current_steps])

请参照 `attention is all you need <https://arxiv.org/pdf/1706.03762.pdf>`_ 

参数
::::::::::::

    - **d_model** (Variable|int) - 模型的输入、输出向量特征维度。类型可设置为标量Tensor，或int值。
    - **warmup_steps** (Variable|int) - 预热步数，类型可设置为标量Tensor，或int值。
    - **learning_rate** (Variable|float|int，可选) - 初始学习率。如果类型为Variable，则为shape为[1]的Tensor，数据类型为float32或float64；也可以是python的int类型。默认值为1.0。

返回
::::::::::::
衰减的学习率

返回类型
::::::::::::
 Variable

代码示例
::::::::::::

.. code-block:: python

        import paddle.fluid as fluid
        warmup_steps = 100
        learning_rate = 0.01
        lr = fluid.layers.learning_rate_scheduler.noam_decay(
                       1/(warmup_steps *(learning_rate ** 2)),
                       warmup_steps,
                       learning_rate)






