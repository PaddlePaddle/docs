.. _cn_api_paddle_optimizer_lr_LRScheduler:

LRScheduler
-----------------------------------

.. py:class:: paddle.optimizer.lr.LRScheduler(learning_rate=0.1, last_epoch=-1, verbose=False) 

学习率策略的基类。定义了所有学习率调整策略的公共接口。

目前在paddle中基于该基类，已经实现了12种策略，分别为：

* :code:`NoamDecay`: 诺姆衰减，相关算法请参考 `《Attention Is All You Need》 <https://arxiv.org/pdf/1706.03762.pdf>`_ 。请参考 :ref:`cn_api_paddle_optimizer_lr_NoamDecay`

* :code:`ExponentialDecay`: 指数衰减，即每次将当前学习率乘以给定的衰减率得到下一个学习率。请参考 :ref:`cn_api_paddle_optimizer_lr_ExponentialDecay`

* :code:`NaturalExpDecay`: 自然指数衰减，即每次将当前学习率乘以给定的衰减率的自然指数得到下一个学习率。请参考 :ref:`cn_api_paddle_optimizer_lr_NaturalExpDecay`

* :code:`InverseTimeDecay`: 逆时间衰减，即得到的学习率与当前衰减次数成反比。请参考 :ref:`cn_api_paddle_optimizer_lr_InverseTimeDecay`

* :code:`PolynomialDecay`: 多项式衰减，即得到的学习率为初始学习率和给定最终学习之间由多项式计算权重定比分点的插值。请参考 :ref:`cn_api_paddle_optimizer_lr_PolynomialDecay`

* :code:`PiecewiseDecay`: 分段衰减，即由给定step数分段呈阶梯状衰减，每段内学习率相同。请参考 :ref:`cn_api_paddle_optimizer_lr_PiecewiseDecay`

* :code:`CosineAnnealingDecay`: 余弦式衰减，即学习率随step数变化呈余弦函数周期变化。请参考 :ref:`cn_api_paddle_optimizer_lr_CosineAnnealingDecay`

* :code:`LinearWarmup`: 学习率随step数线性增加到指定学习率。请参考 :ref:`cn_api_paddle_optimizer_lr_LinearWarmup`

* :code:`StepDecay`: 学习率每隔固定间隔的step数进行衰减，需要指定step的间隔数。请参考 :ref:`cn_api_paddle_optimizer_lr_StepDecay`

* :code:`MultiStepDecay`: 学习率在特定的step数时进行衰减，需要指定衰减时的节点位置。请参考 :ref:`cn_api_paddle_optimizer_lr_MultiStepDecay`

* :code:`LambdaDecay`: 学习率根据自定义的lambda函数进行衰减。请参考 :ref:`cn_api_paddle_optimizer_lr_LambdaDecay`

* :code:`ReduceOnPlateau`: 学习率根据当前监控指标（一般为loss）来进行自适应调整，当loss趋于稳定时衰减学习率。请参考 :ref:`cn_api_paddle_optimizer_lr_ReduceOnPlateau`

你可以继承该基类实现任意的学习率策略，导出基类的方法为 ``form paddle.optimizer.lr import LRScheduler`` ，
必须要重写该基类的 ``get_lr()`` 函数，否则会抛出 ``NotImplementedError`` 异常。

参数：
    - **learning_rate** (float, 可选) - 初始学习率，数据类型为Python float。
    - **last_epoch** (int，可选) - 上一轮的轮数，重启训练时设置为上一轮的epoch数。默认值为 -1，则为初始学习率。
    - **verbose** (bool，可选) - 如果是 ``True`` ，则在每一轮更新时在标准输出 `stdout` 输出一条信息。默认值为 ``False`` 。

返回：用于调整学习率的实例对象。

**代码示例**

这里提供了重载基类 ``LRScheduler`` 并实现 ``StepLR`` 的示例，你可以根据你的需求来实现任意子类。

.. code-block:: python

    import paddle
    from paddle.optimizer.lr import LRScheduler

    class StepDecay(LRScheduler):
        def __init__(self,
                    learning_rate,
                    step_size,
                    gamma=0.1,
                    last_epoch=-1,
                    verbose=False):
            if not isinstance(step_size, int):
                raise TypeError(
                    "The type of 'step_size' must be 'int', but received %s." %
                    type(step_size))
            if gamma >= 1.0:
                raise ValueError('gamma should be < 1.0.')

            self.step_size = step_size
            self.gamma = gamma
            super(StepDecay, self).__init__(learning_rate, last_epoch, verbose)

        def get_lr(self):
            i = self.last_epoch // self.step_size
            return self.base_lr * (self.gamma**i)

.. py:method:: step(epoch=None)

step函数需要在优化器的 `optimizer.step()` 函数之后调用，调用之后将会根据epoch数来更新学习率，更新之后的学习率将会在优化器下一轮更新参数时使用。

参数：
    - **epoch** （int，可选）- 指定具体的epoch数。默认值None，此时将会从-1自动累加 ``epoch`` 数。

返回：无。

**代码示例** ：

请参考 ``基类LRScheduler`` 的任意子类实现，这里以 ``StepLR`` 为例进行了示例：

.. code-block:: python
    
    import paddle
    import numpy as np

    x = np.random.uniform(-1, 1, [10, 10]).astype("float32")
    linear = paddle.nn.Linear(10, 10)
    scheduler = paddle.optimizer.lr.StepDecay(learning_rate=0.5, step_size=5, gamma=0.8, verbose=True)
    sgd = paddle.optimizer.SGD(learning_rate=scheduler, parameters=linear.parameters())
    for epoch in range(20):
        for batch_id in range(2):
            x = paddle.to_tensor(x)
            out = linear(x)
            loss = paddle.mean(out)
            loss.backward()
            sgd.step()
            sgd.clear_gradients()
        scheduler.step()

.. py:method:: get_lr()

如果一个子类继承了 ``基类LRScheduler`` ，则用户必须重写方法 ``get_lr()`` ，否则，将会抛出 ``NotImplementedError`` 异常，

上述给出了实现 ``StepLR`` 的一个简单示例。

.. py:method:: _state_keys()

该函数通过定义字典 ``self.keys`` 来设置 ``optimizer.state_dict()`` 时的存储对象，默认情况下：``self.keys=['last_epoch', 'last_lr']`` ，其中 ``last_epoch``
是当前的epoch数，``last_lr`` 是当前的学习率值。

如果需要改变默认的行为，用户需要重写该方法，来重新定义字典 ``self.keys`` ，一般无需重新设置。
