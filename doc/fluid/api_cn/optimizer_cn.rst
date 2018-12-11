#################
 fluid.optimizer
#################

.. _cn_api_fluid_optimizer_Adadelta:

Adadelta
-------------------------------

.. py:attribute::  paddle.fluid.optimizer.Adadelta

``AdadeltaOptimizer`` 的别名






.. _cn_api_fluid_optimizer_Adagrad:

Adagrad
-------------------------------

.. py:attribute::  paddle.fluid.optimizer.Adagrad

``AdagradOptimizer`` 的别名




.. _cn_api_fluid_optimizer_AdagradOptimizer:

AdagradOptimizer
-------------------------------

.. py:class:: paddle.fluid.optimizer.AdagradOptimizer(learning_rate, epsilon=1e-06, regularization=None, name=None)

**Adaptive Gradient Algorithm(Adagrad)**

更新如下：

.. math::

	moment\_out &= moment + grad * grad\\param\_out 
	&= param - \frac{learning\_rate * grad}{\sqrt{moment\_out} + \epsilon}

原始论文（http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf）没有epsilon属性。在我们的实现中也作了如下更新：
http://cs231n.github.io/neural-networks-3/#ada 用于维持数值稳定性，避免除数为0的错误发生。

参数：
    - **learning_rate** (float|Variable)-学习率，用于更新参数。作为数据参数，可以是一个浮点类型值或者有一个浮点类型值的变量
    - **epsilon** (float) - 维持数值稳定性的短浮点型值
    - **regularization** - 规则化函数，例如fluid.regularizer.L2DecayRegularizer
    - **name** - 名称前缀（可选）

**代码示例**：

.. code-block:: python:

    optimizer = fluid.optimizer.Adagrad(learning_rate=0.2)
    optimizer.minimize(cost)






.. _cn_api_fluid_optimizer_Adam:

Adam
-------------------------------

.. py:attribute::  paddle.fluid.optimizer.Adam

``AdamOptimizer`` 的别名





.. _cn_api_fluid_optimizer_Adamax:

Adamax
-------------------------------

.. py:attribute:: paddle.fluid.optimizer.Adamax

``AdamaxOptimizer`` 的别名






.. _cn_api_fluid_optimizer_AdamaxOptimizer:

AdamaxOptimizer
-------------------------------

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
    目前 ``AdamaxOptimizer`` 不支持  sparse parameter optimization.

  










.. _cn_api_fluid_optimizer_AdamOptimizer:

AdamOptimizer
-------------------------------

.. py:class:: paddle.fluid.optimizer. AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, regularization=None, name=None)

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

**代码示例**：

.. code-block:: python:

    optimizer = fluid.optimizer.Adam(learning_rate=0.2)
    optimizer.minimize(cost)









.. _cn_api_fluid_optimizer_DecayedAdagrad:

DecayedAdagrad
-------------------------------

.. py:attribute::  paddle.fluid.optimizer.DecayedAdagrad

``DecayedAdagradOptimizer`` 的别名





.. _cn_api_fluid_optimizer_DecayedAdagradOptimizer:

DecayedAdagradOptimizer
-------------------------------

.. py:class:: paddle.fluid.optimizer.DecayedAdagradOptimizer(learning_rate, decay=0.95, epsilon=1e-06, regularization=None, name=None)

Decayed Adagrad Optimizer

`原始论文 <http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf>`_

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
  ``DecayedAdagradOptimizer`` 不支持 sparse parameter optimization









.. _cn_api_fluid_optimizer_Ftrl:

Ftrl
-------------------------------

.. py:attribute::  paddle.fluid.optimizer.Ftrl

``FtrlOptimizer`` 的别名




.. _cn_api_fluid_optimizer_FtrlOptimizer:

FtrlOptimizer
-------------------------------

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
  - **l1** (float) - 暂无，请等待后期更新
  - **l2** (float) - 暂无，请等待后期更新
  - **lr_power** (float) - 暂无，请等待后期更新
  - **regularization** - 正则化器，例如 ``fluid.regularizer.L2DecayRegularizer`` 
  - **name** — 可选的名称前缀

抛出异常：
  - ``ValueError`` - 如果 ``learning_rate`` , ``rho`` ,  ``epsilon`` , ``momentum``  为 None.

**代码示例**

.. code-block:: python
        
   optimizer = fluid.optimizer.Ftrl(0.0001)
   _, params_grads = optimizer.minimize(cost)

.. note::
     目前, FtrlOptimizer 不支持 sparse parameter optimization








.. _cn_api_fluid_optimizer_LarsMomentum:

LarsMomentum
-------------------------------

.. py:attribute::  paddle.fluid.optimizer.LarsMomentum

``fluid.optimizer.LarsMomentumOptimizer`` 的别名





.. _cn_api_fluid_optimizer_LarsMomentumOptimizer:

LarsMomentumOptimizer
-------------------------------

.. py:function:: paddle.fluid.optimizer.LarsMomentumOptimizer(learning_rate, momentum, lars_coeff=0.001, lars_weight_decay=0.0005, regularization=None, name=None)

LARS支持的Momentum优化器

公式作如下更新：

.. math::

  & local\_learning\_rate = learning\_rate * lars\_coeff * \
  \frac{||param||}{||gradient|| + lars\_weight\_decay * ||param||}\\
  & velocity = mu * velocity + local\_learning\_rate * (gradient + lars\_weight\_decay * param)\\
  & param = param - velocity

参数：
  - **learning_rate** (float|Variable) - 学习率，用于参数更新。作为数据参数，可以是浮点型值或含有一个浮点型值的变量
  - **momentum** (float) - 动量因子
  - **lars_coeff** (float) - 定义LARS本地学习率的权重
  - **lars_weight_decay** (float) - 使用LARS进行衰减的权重衰减系数
  - **regularization** - 正则化函数，例如 :code:`fluid.regularizer.L2DecayRegularizer`
  - **name** - 名称前缀，可选

**代码示例：**

.. code-block:: python

    optimizer = fluid.optimizer.LarsMomentum(learning_rate=0.2, momentum=0.1, lars_weight_decay=0.001)
    optimizer.minimize(cost)







.. _cn_api_fluid_optimizer_ModelAverage:

ModelAverage
-------------------------------

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








.. _cn_api_fluid_optimizer_Momentum:

Momentum
-------------------------------

.. py:attribute::  paddle.fluid.optimizer.Momentum

``MomentumOptimizer`` 的别名



.. _cn_api_fluid_optimizer_MomentumOptimizer:

MomentumOptimizer
-------------------------------

.. py:class::  paddle.fluid.optimizer.MomentumOptimizer(learning_rate, momentum, use_nesterov=False, regularization=None, name=None)

含有速度状态的Simple Momentum 优化器

该优化器含有牛顿动量标志，公式更新如下：

.. math::
	& velocity = mu * velocity + gradient\\
	& if (use\_nesterov):\
	\&\quad   param = param - (gradient + mu * velocity) * learning\_rate\\
	& else:\\&\quad   param = param - learning\_rate * velocity
参数：
    - **learning_rate** (float|Variable) - 学习率，用于参数更新。作为数据参数，可以是浮点型值或含有一个浮点型值的变量
    - **momentum** (float) - 动量因子
    - **use_nesterov** (bool) - 赋能牛顿动量
    - **regularization** - 正则化函数，比如fluid.regularizer.L2DecayRegularizer
    - **name** - 名称前缀（可选）

**代码示例**：

.. code-block:: python

    optimizer = fluid.optimizer.Momentum(learning_rate=0.2, momentum=0.1)
    optimizer.minimize(cost)







.. _cn_api_fluid_optimizer_RMSPropOptimizer:

RMSPropOptimizer
-------------------------------

.. py:class:: paddle.fluid.optimizer.RMSPropOptimizer(learning_rate, rho=0.95, epsilon=1e-06, momentum=0.0, centered=False, regularization=None, name=None)

均方根平均传播（RMSProp）法是一种未发表的,自适应学习率的方法。原始slides提出了RMSProp：[http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf]中的第29张slide。等式如下所示：

.. math::
    r(w, t) & = \rho r(w, t-1) + (1 - \rho)(\nabla Q_{i}(w))^2\\
    w & = w - \frac{\eta} {\sqrt{r(w,t) + \epsilon}} \nabla Q_{i}(w)
    
第一个等式计算每个权重平方梯度的移动平均值，然后将梯度除以 :math:`sqrtv（w，t）` 。
  
.. math::
   r(w, t) & = \rho r(w, t-1) + (1 - \rho)(\nabla Q_{i}(w))^2\\
   v(w, t) & = \beta v(w, t-1) +\frac{\eta} {\sqrt{r(w,t) +\epsilon}} \nabla Q_{i}(w)\\
         w & = w - v(w, t)

如果居中为真：
  
.. math::
      r(w, t) & = \rho r(w, t-1) + (1 - \rho)(\nabla Q_{i}(w))^2\\
      g(w, t) & = \rho g(w, t-1) + (1 -\rho)\nabla Q_{i}(w)\\
      v(w, t) & = \beta v(w, t-1) + \frac{\eta} {\sqrt{r(w,t) - (g(w, t))^2 +\epsilon}} \nabla Q_{i}(w)\\
            w & = w - v(w, t)
      
其中， :math:`ρ` 是超参数，典型值为0.9,0.95等。 :math:`beta` 是动量术语。  :math:`epsilon` 是一个平滑项，用于避免除零，通常设置在1e-4到1e-8的范围内。
      
参数：
    - **learning_rate** （float） - 全球学习率。
    - **rho** （float） - rho是等式中的 :math:`rho` ，默认设置为0.95。
    - **epsilon** （float） - 等式中的epsilon是平滑项，避免被零除，默认设置为1e-6。
    - **momentum** （float） - 方程中的β是动量项，默认设置为0.0。
    - **centered** （bool） - 如果为True，则通过梯度估计方差对梯度进行归一化；如果false，则由未centered的第二个moment归一化。将此设置为True有助于培训，但在计算和内存方面稍微昂贵一些。默认为False。
    - **regularization**  - 正则器项，如 ``fluid.regularizer.L2DecayRegularizer`` 。
    - **name**  - 可选的名称前缀。
    
抛出异常:
    - ``ValueError`` -如果 ``learning_rate`` ， ``rho`` ， ``epsilon`` ， ``momentum`` 为None。

**示例代码**

..  code-block:: python

        optimizer = fluid.optimizer.RMSProp(0.0001)
        _, params_grads = optimizer.minimize(cost)










.. _cn_api_fluid_optimizer_SGD:

SGD
-------------------------------

.. py:attribute::  paddle.fluid.optimizer.SGD

``SGDOptimizer`` 的别名






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
        
     sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.2)
     sgd_optimizer.minimize(cost)









