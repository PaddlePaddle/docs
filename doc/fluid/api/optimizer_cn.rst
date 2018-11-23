


.. _cn_api_fluid_optimizer_SGDOptimizer:

SGDOptimizer
>>>>>>>>>>>>

.. py:class:: paddle.fluid.optimizer.SGDOptimizer(learning_rate, regularization=None, name=None)

随机梯度下降算法的优化器

.. math::
            \\param\_out=param-learning\_rate*grad\\


参数:

  - **learning_rate** (float|Variable) - 用于更新参数的学习率。可以是浮点值，也可以是具有一个浮点值作为数据元素的变量。

  - **regularization** - 一个正则化器，例如 ``fluid.regularizer.L2DecayRegularizer`` 

  - **name** — 可选的名称前缀。
  
  
**代码示例**
 
.. code-block:: python
        
     sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.2)
     sgd_optimizer.minimize(cost)



.. _cn_api_fluid_optimizer_AdamaxOptimizer:

AdamaxOptimizer
>>>>>>>>>>>>

.. py:class:: paddle.fluid.optimizer.AdamaxOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, regularization=None, name=None)

我们参考Adam论文第7节中的Adamax优化: https://arxiv.org/abs/1412.6980 ， Adamax是基于无穷大范数的Adam算法的一个变种。


Adamax 更新规则:

.. math::
    \\t = t + 1
.. math::
    moment\_out=\beta_1∗moment+(1−\beta_1)∗grad
.. math::
    inf\_norm\_out=\max{(\beta_2∗inf\_norm+ϵ, \left|grad\right|)}
.. math::
    learning\_rate=\frac{learning\_rate}{1-\beta_1^t}
.. math::
    param\_out=param−learning\_rate*\frac{moment\_out}{inf\_norm\_out}\\


论文中没有 ``epsilon`` 参数。但是，为了数值稳定性， 防止除0错误， 增加了这个参数

参数:
  - **learning_rate**  (float|Variable) - 用于更新参数的学习率。可以是浮点值，也可以是具有一个浮点值作为数据元素的变量。
  - **beta1** (float) - 第1阶段估计的指数衰减率
  - **beta2** (float) - 第2阶段估计的指数衰减率。
  - **epsilon** (float) -非常小的浮点值，为了数值的稳定性质
  - **regularization** - 正则化器，例如 ``fluid.regularizer.L2DecayRegularizer`` 
  - **name** - 可选的名称前缀。

**代码示例**
 
.. code-block:: python
        
     optimizer = fluid.optimizer.Adamax(learning_rate=0.2)
     optimizer.minimize(cost)

.. note::
    目前 ``AdamaxOptimizer`` 不支持  sparse gradient

  
.. _cn_api_fluid_optimizer_DecayedAdagradOptimizer:

DecayedAdagradOptimizer
>>>>>>>>>>>>

.. py:class:: paddle.fluid.optimizer.DecayedAdagradOptimizer(learning_rate, decay=0.95, epsilon=1e-06, regularization=None, name=None)

Decayed Adagrad Optimizer

原始论文： `http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf <http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf>`_ 


原始论文： `http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf <http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf>`_  中没有 ``epsilon`` 参数。但是，为了数值稳定性， 防止除0错误， 增加了这个参数

.. math::
    moment\_out = decay*moment+(1-decay)*grad*grad
.. math::
    param\_out=param-\frac{learning\_rate*grad}{\sqrt{moment\_out+\epsilon }}
    
参数:

  - **learning_rate** (float|Variable) - 用于更新参数的学习率。可以是浮点值，也可以是具有一个浮点值作为数据元素的变量。
  - **decay** (float) – 衰减率
  - **regularization** - 一个正则化器，例如 ``fluid.regularizer.L2DecayRegularizer`` 
  - **epsilon** (float) - 非常小的浮点值，为了数值稳定性
  - **name** — 可选的名称前缀。

  
**代码示例**
 
.. code-block:: python
        
  optimizer = fluid.optimizer.DecayedAdagrad(learning_rate=0.2)
  optimizer.minimize(cost)

.. note::
  ``DecayedAdagradOptimizer`` 不支持 sparse gradient


.. _cn_api_fluid_optimizer_FtrlOptimizer:

FtrlOptimizer
>>>>>>>>>>>>

.. py:class:: paddle.fluid.optimizer.FtrlOptimizer(learning_rate, l1=0.0, l2=0.0, lr_power=-0.5,regularization=None, name=None)
 
FTRL (Follow The Regularized Leader) Optimizer.

TFRTL 原始论文: ( `https://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf <https://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf>`_)


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
  - **l1** (float)
  - **l2** (float)
  - **lr_power** (float)
  - **regularization** - 正则化器，例如 ``fluid.regularizer.L2DecayRegularizer`` 
  - **name** — 可选的名称前缀

抛出异常：
  - ``ValueError``  如果 ``learning_rate`` , ``rho`` ,  ``epsilon`` , ``momentum``  为 None.

**代码示例**

.. code-block:: python
        
   optimizer = fluid.optimizer.Ftrl(0.0001)
   _, params_grads = optimizer.minimize(cost)
   
目前, FtrlOptimizer 不支持 sparse gradient


.. _cn_api_fluid_optimizer_ModelAverage:

ModelAverage
>>>>>>>>>>>>

.. py:class:: paddle.fluid.optimizer.ModelAverage(average_window_rate, min_average_window=10000, max_average_window=10000, regularization=None, name=None)

在滑动窗口中累积参数的平均值。平均结果将保存在临时变量中，通过调用 ``apply()`` 方法可应用于当前模型的参数变量。使用 ``restore()`` 方法恢复当前模型的参数值。

平均窗口的大小由 ``average_window_rate`` ， ``min_average_window`` ， ``max_average_window`` 以及当前更新次数决定。

 
参数:
  - **average_window_rate** – 窗口平均速率
  - **min_average_window** – 平均窗口大小的最小值
  - **max_average_window** – 平均窗口大小的最大值
  - **regularization** – 正则化器，例如 ``fluid.regularizer.L2DecayRegularizer`` 
  - **name** – 可选的名称前缀

**代码示例**

.. code-block:: python
        
  optimizer = fluid.optimizer.Momentum()
  optimizer.minimize(cost)
  model_average = fluid.optimizer.ModelAverage(0.15,
                                          min_average_window=10000,
                                          max_average_window=20000)
  for pass_id in range(args.pass_num):
      for data in train_reader():
          exe.run(fluid.default_main_program()...)

      with model_average.apply(exe):
          for data in test_reader():
              exe.run(inference_program...)


.. py:method:: apply(*args, **kwds)

将平均值应用于当前模型的参数。

.. py:method:: restore(executor)

恢复当前模型的参数值

