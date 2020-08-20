.. _cn_api_optimizer_ExponentialLR:

ExponentialLR
-----------------------------------

.. py:class:: paddle.optimizer.ExponentialLR(learning_rate, gamma, last_epoch=-1, verbose=False)

该接口提供一种学习率按指数函数衰减的功能。

衰减函数可以用以下公式表示：

.. math::

 current\_lr = learning\_rate * \gamma^{epoch}

式子中，

- **epoch** ：训练轮数。
- **current_lr** ： 当前学习率。

每一轮使用衰减率 **gamma** 衰减学习率。当 **last_epoch** 为-1时，设置初始学习率为learning_rate。

参数
:::::::::
    - **learning_rate** （float|int）：初始学习率，可以是Python的float或int。
    - **gamma** （float）：衰减率。
    - **last_epoch** （int）：上一轮的下标。默认为 `-1` 。
    - **verbose** （bool）：如果是 `True` ，则在每一轮更新时在标准输出 `stdout` 输出一条信息。

返回
:::::::::
    无

代码示例
:::::::::

.. code-block:: python


