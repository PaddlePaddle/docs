.. _cn_api_paddle_optimizer_lr_LRScheduler:

LRScheduler
-----------------------------------

.. py:class:: paddle.optimizer.lr.LRScheduler(learning_rate=0.1, last_epoch=-1, verbose=False)

学习率策略的基类。定义了所有学习率调整策略的公共接口。

目前在 paddle 中基于该基类，已经实现了 14 种策略，分别为：

* :code:`NoamDecay`：诺姆衰减，相关算法请参考 `《Attention Is All You Need》 <https://arxiv.org/pdf/1706.03762.pdf>`_ 。请参考 :ref:`cn_api_paddle_optimizer_lr_NoamDecay`。

* :code:`ExponentialDecay`：指数衰减，即每次将当前学习率乘以给定的衰减率得到下一个学习率。请参考 :ref:`cn_api_paddle_optimizer_lr_ExponentialDecay`。

* :code:`NaturalExpDecay`：自然指数衰减，即每次将当前学习率乘以给定的衰减率的自然指数得到下一个学习率。请参考 :ref:`cn_api_paddle_optimizer_lr_NaturalExpDecay`。

* :code:`InverseTimeDecay`：逆时间衰减，即得到的学习率与当前衰减次数成反比。请参考 :ref:`cn_api_paddle_optimizer_lr_InverseTimeDecay`。

* :code:`PolynomialDecay`：多项式衰减，即得到的学习率为初始学习率和给定最终学习之间由多项式计算权重定比分点的插值。请参考 :ref:`cn_api_paddle_optimizer_lr_PolynomialDecay`。

* :code:`PiecewiseDecay`：分段衰减，即由给定 step 数分段呈阶梯状衰减，每段内学习率相同。请参考 :ref:`cn_api_paddle_optimizer_lr_PiecewiseDecay`。

* :code:`CosineAnnealingDecay`：余弦式衰减，即学习率随 step 数变化呈余弦函数周期变化。请参考 :ref:`cn_api_paddle_optimizer_lr_CosineAnnealingDecay`。

* :code:`LinearWarmup`：学习率随 step 数线性增加到指定学习率。请参考 :ref:`cn_api_paddle_optimizer_lr_LinearWarmup`。

* :code:`StepDecay`：学习率每隔固定间隔的 step 数进行衰减，需要指定 step 的间隔数。请参考 :ref:`cn_api_paddle_optimizer_lr_StepDecay`。

* :code:`MultiStepDecay`：学习率在特定的 step 数时进行衰减，需要指定衰减时的节点位置。请参考 :ref:`cn_api_paddle_optimizer_lr_MultiStepDecay`。

* :code:`LambdaDecay`：学习率根据自定义的 lambda 函数进行衰减。请参考 :ref:`cn_api_paddle_optimizer_lr_LambdaDecay`。

* :code:`ReduceOnPlateau`：学习率根据当前监控指标（一般为 loss）来进行自适应调整，当 loss 趋于稳定时衰减学习率。请参考 :ref:`cn_api_paddle_optimizer_lr_ReduceOnPlateau`。

* :code:`MultiplicativeDecay`：每次将当前学习率乘以 lambda 函数得到下一个学习率。请参考 :ref:`cn_api_paddle_optimizer_lr_MultiplicativeDecay`。

* :code:`OneCycleLR`: One Cycle 衰减，学习率上升至最大，再下降至最小。请参考 :ref:`cn_api_paddle_optimizer_lr_OneCycleLR`。

* :code:`CyclicLR`: Cyclic 学习率衰减，其将学习率变化的过程视为一个又一个循环，学习率根据固定的频率在最小和最大学习率之间不停变化。请参考 :ref:`cn_api_paddle_optimizer_lr_CyclicLR`。

你可以继承该基类实现任意的学习率策略，导出基类的方法为 ``from paddle.optimizer.lr import LRScheduler`` ，
必须要重写该基类的 ``get_lr()`` 函数，否则会抛出 ``NotImplementedError`` 异常。

参数
::::::::::::

    - **learning_rate** (float，可选) - 初始学习率，数据类型为 Python float。
    - **last_epoch** (int，可选) - 上一轮的轮数，重启训练时设置为上一轮的 epoch 数。默认值为 -1，则为初始学习率。
    - **verbose** (bool，可选) - 如果是 ``True``，则在每一轮更新时在标准输出 `stdout` 输出一条信息。默认值为 ``False`` 。

返回
::::::::::::
用于调整学习率的实例对象。

代码示例
::::::::::::

COPY-FROM: paddle.optimizer.lr.LRScheduler

方法
::::::::::::
step(epoch=None)
'''''''''

step 函数需要在优化器的 `optimizer.step()` 函数之后调用，调用之后将会根据 epoch 数来更新学习率，更新之后的学习率将会在优化器下一轮更新参数时使用。

**参数**

    - **epoch** （int，可选）- 指定具体的 epoch 数。默认值 None，此时将会从-1 自动累加 ``epoch`` 数。

**返回**

无。

**代码示例**

COPY-FROM: paddle.optimizer.lr.LRScheduler.step

get_lr()
'''''''''

如果一个子类继承了 ``基类 LRScheduler``，则用户必须重写方法 ``get_lr()``，否则，将会抛出 ``NotImplementedError`` 异常，

上述给出了实现 ``StepLR`` 的一个简单示例。

_state_keys()
'''''''''

该函数通过定义字典 ``self.keys`` 来设置 ``optimizer.state_dict()`` 时的存储对象，默认情况下：``self.keys=['last_epoch', 'last_lr']``，其中 ``last_epoch``
是当前的 epoch 数，``last_lr`` 是当前的学习率值。

如果需要改变默认的行为，用户需要重写该方法，来重新定义字典 ``self.keys``，一般无需重新设置。
