.. _cn_api_fluid_optimizer_AdamOptimizer:

AdamOptimizer
-------------------------------

.. py:class:: paddle.fluid.optimizer.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, regularization=None, name=None, lazy_mode=False)

该函数实现了自适应矩估计优化器，介绍自 `Adam论文 <https://arxiv.org/abs/1412.6980>`_ 的第二节。Adam是一阶基于梯度下降的算法，基于自适应低阶矩估计。
Adam更新如下：

.. math::

    t & = t + 1\\moment\_out & = {\beta}_1 * moment + (1 - {\beta}_1) * grad\\inf\_norm\_out & = max({\beta}_2 * inf\_norm + \epsilon, |grad|)\\learning\_rate & = \frac{learning\_rate}{1 - {\beta}_1^t}\\param\_out & = param - learning\_rate * \frac{moment\_out}{inf\_norm\_out}

参数: 
    - **learning_rate** (float|Variable)-学习率，用于更新参数。作为数据参数，可以是一个浮点类型值或有一个浮点类型值的变量
    - **beta1** (float)-一阶矩估计的指数衰减率
    - **beta2** (float)-二阶矩估计的指数衰减率
    - **epsilon** (float)-保持数值稳定性的短浮点类型值
    - **regularization** - 规则化函数，例如''fluid.regularizer.L2DecayRegularizer
    - **name** - 可选名称前缀
    - **lazy_mode** （bool: false） - 官方Adam算法有两个移动平均累加器（moving-average accumulators）。累加器在每一步都会更新。在密集模式和稀疏模式下，两条移动平均线的每个元素都会更新。如果参数非常大，那么更新可能很慢。 lazy mode仅更新当前具有梯度的元素，所以它会更快。但是这种模式与原始的算法有不同的描述，可能会导致不同的结果。


**代码示例**：

.. code-block:: python:

    import paddle
    import paddle.fluid as fluid
     
    place = fluid.CPUPlace()
    main = fluid.Program()
    with fluid.program_guard(main):
        x = fluid.layers.data(name='x', shape=[13], dtype='float32')
        y = fluid.layers.data(name='y', shape=[1], dtype='float32')
        y_predict = fluid.layers.fc(input=x, size=1, act=None)
        cost = fluid.layers.square_error_cost(input=y_predict, label=y)
        avg_cost = fluid.layers.mean(cost)
        adam_optimizer = fluid.optimizer.AdamOptimizer(0.01)
        adam_optimizer.minimize(avg_cost)

        fetch_list = [avg_cost]
        train_reader = paddle.batch(
            paddle.dataset.uci_housing.train(), batch_size=1)
        feeder = fluid.DataFeeder(place=place, feed_list=[x, y])
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
        for data in train_reader():
            exe.run(main, feed=feeder.feed(data), fetch_list=fetch_list)








