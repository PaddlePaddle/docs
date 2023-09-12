.. _cn_api_paddle_static_ExponentialMovingAverage:

ExponentialMovingAverage
-------------------------------

.. py:class:: paddle.static.ExponentialMovingAverage(decay=0.999, thres_steps=None, name=None)

用指数衰减计算参数的滑动平均值。给定参数 :math:`\theta`，它的指数滑动平均值 (exponential moving average, EMA) 为

.. math::
    \begin{align}\begin{aligned}\text{EMA}_0 & = 0\\\text{EMA}_t & = \text{decay} * \text{EMA}_{t-1} + (1 - \text{decay}) * \theta_t\end{aligned}\end{align}

用 ``update()`` 方法计算出的平均结果将保存在由实例化对象创建和维护的临时变量中，并且可以通过调用 ``apply()`` 方法把结果应用于当前模型的参数。同时，可用 ``restore()`` 方法恢复原始参数。

**偏置校正**

所有的滑动平均均初始化为 :math:`0`，因此它们相对于零是有偏的，可以通过除以因子 :math:`(1 - \text{decay}^t)` 来校正，因此在调用 ``apply()`` 方法时，作用于参数的真实滑动平均值将为：

.. math::
    \widehat{\text{EMA}}_t = \frac{\text{EMA}_t}{1 - \text{decay}^t}

**衰减率调节**

一个非常接近于 1 的很大的衰减率将会导致平均值滑动得很慢。更优的策略是，开始时设置一个相对较小的衰减率。参数 ``thres_steps`` 允许用户传递一个变量以设置衰减率，在这种情况下，
真实的衰减率变为：

.. math::
    \min(\text{decay}, \frac{1 + \text{thres_steps}}{10 + \text{thres_steps}})
通常 ``thres_steps`` 可以是全局的训练迭代步数。

参数
:::::::::
    - **decay** (float，可选) – 指数衰减率，通常接近 1，如 0.999 ，0.9999 ，···。默认值为 0.999 。
    - **thres_steps** (Variable|None，可选) – 调节衰减率的阈值步数，默认值为 None 。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

代码示例
:::::::::
COPY-FROM: paddle.static.ExponentialMovingAverage


方法
::::::::::::
update()
'''''''''

更新指数滑动平均，在训练过程中需调用此方法。


apply(executor, need_restore=True)
'''''''''

模型评测时，将滑动平均的结果作用在参数上。

**参数**

    - **executor** (Executor) – 将滑动平均值作用在参数上的执行器。
    - **need_restore** (bool，可选) – 是否在结束后恢复原始参数，默认值为 ``True`` 。


restore(executor)
'''''''''

恢复参数。

**参数**

    - **executor** (Executor) – 执行恢复动作的执行器。
