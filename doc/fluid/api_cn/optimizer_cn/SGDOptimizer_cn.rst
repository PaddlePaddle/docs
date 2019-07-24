.. _cn_api_fluid_optimizer_SGDOptimizer:

SGDOptimizer
-------------------------------

.. py:class:: paddle.fluid.optimizer.SGDOptimizer(learning_rate, regularization=None, name=None)

随机梯度下降算法的优化器

.. math::
            \\param\_out=param-learning\_rate*grad\\


参数:
  - **learning_rate** (float|Variable) - 用于更新参数的学习率。可以是浮点值，也可以是具有一个浮点值作为数据元素的变量。
  - **regularization** - 一个正则化器，例如 ``fluid.regularizer.L2DecayRegularizer`` 
  - **name** - 可选的名称前缀。
  
  
**代码示例**
 
.. code-block:: python
    
    import paddle
    import paddle.fluid as fluid
    import numpy as np
     
    place = fluid.CPUPlace()
    main = fluid.Program()
    with fluid.program_guard(main):
        x = fluid.layers.data(name='x', shape=[13], dtype='float32')
        y = fluid.layers.data(name='y', shape=[1], dtype='float32')
        y_predict = fluid.layers.fc(input=x, size=1, act=None)
        cost = fluid.layers.square_error_cost(input=y_predict, label=y)
        avg_cost = fluid.layers.mean(cost)   
        
        sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.001)
        sgd_optimizer.minimize(avg_cost)

        fetch_list = [avg_cost]
        train_reader = paddle.batch(
            paddle.dataset.uci_housing.train(), batch_size=1)
        feeder = fluid.DataFeeder(place=place, feed_list=[x, y])
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
        for data in train_reader():
            exe.run(main, feed=feeder.feed(data), fetch_list=fetch_list)









