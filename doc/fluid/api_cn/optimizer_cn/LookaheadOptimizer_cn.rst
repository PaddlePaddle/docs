.. _cn_api_fluid_optimizer_LookaheadOptimizer:

LookaheadOptimizer
-------------------------------

.. py:class:: paddle.fluid.optimizer.LookaheadOptimizer(inner_optimizer, alpha=0.5, k=5)

本类实现了Lookahead优化算法：https://arxiv.org/abs/1907.08610。Lookahead优化算法在内存中保存两部分参数：快参数和慢参数。每个训练步次，inner_optimizer都更新快参数；每隔k个训练步次，Lookahead更新慢参数，如下：

.. math::

  & slow\_param_t = slow\_param_{t-1} + \alpha * (fast\_param_{t-1} - slow\_param_{t-1})

  & fast\_param_t = slow\_param_t

参数:
    - **inner_optimizer** (Optimizer) - 基础优化器，如SGD
    - **alpha** (float) - Lookahead 的学习率
    - **k** (int) - 慢参数更新的频率：k次一更新

**代码示例**

.. code-block:: python

            import paddle
            import paddle.fluid as fluid
            import numpy as np

            x = fluid.layers.data(name='x', shape=[2], dtype='float32')
            label = fluid.layers.data(name="label", shape=[1], dtype="int64")
            y = fluid.layers.fc(input=[x], size=2, act="softmax")
            loss = fluid.layers.cross_entropy(input=y, label=label)
            loss = fluid.layers.mean(x=loss)
            sgd = fluid.optimizer.SGD(learning_rate=0.01)
            optimizer = fluid.optimizer.LookaheadOptimizer(sgd,
                                            alpha=0.5,
                                            k=5)
            optimizer.minimize(loss)
            main_program = fluid.default_main_program()
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())

            feeder = fluid.DataFeeder(feed_list=[x, label], place=place)

            step = 0
            while(step < 10):
                step += 1
                exe.run(fluid.default_main_program(),
                feed=feeder.feed(batch_data))

