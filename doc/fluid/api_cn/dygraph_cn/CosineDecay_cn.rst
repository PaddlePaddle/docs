.. _cn_api_fluid_dygraph_CosineDecay:

CosineDecay
-------------------------------

.. py:class:: paddle.fluid.dygraph.CosineDecay(learning_rate, step_each_epoch, epochs, begin=0, step=1, dtype='float32')

使用 cosine decay 的衰减方式进行学习率调整。

在训练模型时，建议一边进行训练一边降低学习率。 通过使用此方法，学习率将通过如下cosine衰减策略进行衰减：

.. math::

    decayed\_lr = learning\_rate * 0.5 * (math.cos * (epoch * \frac{math.pi}{epochs} ) + 1)


参数：
    - **learning_rate** (Variable | float) - 初始学习率。
    - **step_each_epoch** （int） - 一次迭代中的步数。
    - **begin** (int) - 起始步，默认为0。
    - **step** (int) - 步大小，默认为1。
    - **dtype**  (str) - 学习率的dtype，默认为‘float32’


**代码示例**

.. code-block:: python

    base_lr = 0.1
    with fluid.dygraph.guard():
        optimizer  = fluid.optimizer.SGD(
            learning_rate = fluid.dygraph.CosineDecay(
                    base_lr, 10000, 120) )




