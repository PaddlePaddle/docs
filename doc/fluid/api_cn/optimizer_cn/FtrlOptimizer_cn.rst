.. _cn_api_fluid_optimizer_FtrlOptimizer:

FtrlOptimizer
-------------------------------

.. py:class:: paddle.fluid.optimizer.FtrlOptimizer(learning_rate, l1=0.0, l2=0.0, lr_power=-0.5,regularization=None, name=None)
 
FTRL (Follow The Regularized Leader) Optimizer.

FTRL 原始论文: ( `https://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf <https://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf>`_)


.. math::
           &\qquad new\_accum=squared\_accum+grad^2\\\\
           &\qquad if(lr\_power==−0.5):\\
           &\qquad \qquad linear\_accum+=grad-\frac{\sqrt{new\_accum}-\sqrt{squared\_accum}}{learning\_rate*param}\\
           &\qquad else:\\
           &\qquad \qquad linear\_accum+=grad-\frac{new\_accum^{-lr\_power}-accum^{-lr\_power}}{learning\_rate*param}\\\\
           &\qquad x=l1*sign(linear\_accum)−linear\_accum\\\\
           &\qquad if(lr\_power==−0.5):\\
           &\qquad \qquad y=\frac{\sqrt{new\_accum}}{learning\_rate}+(2*l2)\\
           &\qquad \qquad pre\_shrink=\frac{x}{y}\\
           &\qquad \qquad param=(abs(linear\_accum)>l1).select(pre\_shrink,0.0)\\
           &\qquad else:\\
           &\qquad \qquad y=\frac{new\_accum^{-lr\_power}}{learning\_rate}+(2*l2)\\
           &\qquad \qquad pre\_shrink=\frac{x}{y}\\
           &\qquad \qquad param=(abs(linear\_accum)>l1).select(pre\_shrink,0.0)\\\\
           &\qquad squared\_accum+=grad^2


参数:
  - **learning_rate** (float|Variable)-全局学习率。
  - **l1** (float) - L1 regularization strength.
  - **l2** (float) - L2 regularization strength.
  - **lr_power** (float) - 学习率降低指数
  - **regularization** - 正则化器，例如 ``fluid.regularizer.L2DecayRegularizer`` 
  - **name** — 可选的名称前缀

抛出异常：
  - ``ValueError`` - 如果 ``learning_rate`` , ``rho`` ,  ``epsilon`` , ``momentum``  为 None.

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
    
        ftrl_optimizer = fluid.optimizer.Ftrl(learning_rate=0.1)
        ftrl_optimizer.minimize(avg_cost)
    
        fetch_list = [avg_cost]
        train_reader = paddle.batch(
            paddle.dataset.uci_housing.train(), batch_size=1)
        feeder = fluid.DataFeeder(place=place, feed_list=[x, y])
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
        for data in train_reader():
            exe.run(main, feed=feeder.feed(data), fetch_list=fetch_list)


.. note::
     目前, FtrlOptimizer 不支持 sparse parameter optimization




