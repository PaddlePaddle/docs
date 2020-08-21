.. _cn_api_fluid_optimizer_NaturalExpLR:

NaturalExpLR
-------------------------------


.. py:class:: paddle.optimizer.lr_scheduler.NaturalExpLR(learning_rate, gama, last_epoch=-1, verbose=False)

该接口提供按自然指数衰减学习率的功能。

自然指数衰减的计算方式如下。

.. math::

    decayed\_learning\_rate = learning\_rate * e^{y} 

当staircase为False时，y对应的计算公式为：

.. math::

    y = - decay\_rate * \frac{global\_step}{decay\_steps}

当staircase为True时，y对应的计算公式为：

.. math::

    y = - decay\_rate * math.floor(\frac{global\_step}{decay\_steps}) 

式中，

- :math:`decayed\_learning\_rate` ： 衰减后的学习率。
式子中各参数详细介绍请看参数说明。

参数
:::::::::
    - **learning_rate** (Tensor|float) - 初始学习率值。如果类型为Tensor，则为shape为[1]的Tensor，数据类型为float32或float64；也可以是python的float类型。
    - **gamma** （float）：衰减率。
    - **last_epoch** （int，可选）: 上一轮的轮数，重启训练时设置为上一轮的epoch数。默认值为 -1，则为初始学习率。
    - **verbose** （bool，可选）：如果是 `True` ，则在每一轮更新时在标准输出 `stdout` 输出一条信息。

返回
:::::::::
返回计算NaturalExpLR的可调用对象。

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

