.. _cn_api_fluid_dygraph_CosineDecay:

CosineDecay
-------------------------------

.. py:class:: paddle.fluid.dygraph.CosineDecay(learning_rate, step_each_epoch, epochs, begin=0, step=1, dtype='float32')

该接口提供学习率按cosine函数周期衰减的功能。

计算方式如下。

.. math::

    decayed\_lr = learning\_rate * 0.5 * (math.cos * (epoch * \frac{math.pi}{epochs} ) + 1)


参数：
    - **learning_rate** (Variable|float) - 初始学习率。类型可以为学习率变量(Variable)或python float常量。
    - **step_each_epoch** （int） - 遍历一遍训练数据所需的步数。
    - **epochs** （int） - 遍历训练数据的轮数。
    - **begin** (int) - 起始步。默认值为0。
    - **step** (int) - 步大小。默认值为1。
    - **dtype**  (str) - 初始化学习率变量的dtype。默认值为'float32'。


**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    base_lr = 0.1
    with fluid.dygraph.guard():
        optimizer  = fluid.optimizer.SGD(
            learning_rate = fluid.dygraph.CosineDecay(
                    base_lr, 10000, 120) )




