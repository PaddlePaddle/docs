.. _cn_api_fluid_optimizer_NoamDecay:

NoamDecay
-------------------------------


.. py:class:: paddle.optimizer.lr_scheduler.NoamLR(d_model, warmup_steps, learning_rate=1.0, last_epoch=-1, verbose=False)


该接口提供Noam衰减学习率的功能。

Noam衰减的计算方式如下。

.. math::

    decayed\_learning\_rate = learning\_rate * d_{model}^{-0.5} * min(global\_steps^{-0.5}, global\_steps * warmup\_steps^{-1.5})

关于Noam衰减的更多细节请参考 `attention is all you need <https://arxiv.org/pdf/1706.03762.pdf>`_

式中，

- :math:`decayed\_learning\_rate` ： 衰减后的学习率。
式子中各参数详细介绍请看参数说明。

参数
:::::::::
    - **d$_{model}$**  (Tensor|int) - 模型的输入、输出向量特征维度，为超参数。如果设置为Tensor类型值，则数据类型可以为int32或int64的标量Tensor，也可以设置为Python int。
    - **warmup_steps** (Tensor|int) - 预热步数，为超参数。如果设置为Tensor类型，则数据类型为int32或int64的标量Tensor，也可以设置为为Python int。
    - **learning_rate** (Tensor|float|int，可选) - 初始学习率。如果类型为Tensor，则为shape为[1]的Tensor，数据类型为float32或float64；也可以是python的int类型。默认值为1.0。
    - **last_epoch** （int，可选）: 上一轮的轮数，重启训练时设置为上一轮的epoch数。默认值为 -1，则为初始学习率。
    - **verbose** （bool，可选）：如果是 `True` ，则在每一轮更新时在标准输出 `stdout` 输出一条信息。

返回
:::::::::
返回计算InverseTimeLR的可调用对象。

代码示例
:::::::::

.. code-block:: python




.. py:method:: step(epoch)

通过当前的 ``step`` 函数调整学习率，调整后的学习率将会在下一个step生效。

参数：
  - **step** （float|int，可选）- 类型：int或float。当前的step数。默认：无，此时将会自动累计 ``step`` 数。

返回：
  无。

**代码示例** ：

  参照上述示例代码。


