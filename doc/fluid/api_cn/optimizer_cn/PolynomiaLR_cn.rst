.. _cn_api_fluid_optimizer_PolynomialLR:

PolynomialLR
-------------------------------


.. py:class:: paddle.optimizer.lr_scheduler.PolynomialLR(learning_rate, gama, end_lr=0.0001, power=1.0, cycle=False, last_epoch=-1, verbose=False)


该接口提供学习率按多项式衰减的功能。通过多项式衰减函数，使得学习率值逐步从初始的 ``learning_rate``，衰减到 ``end_learning_rate`` 。

计算方式如下。

若cycle为True，则计算公式为：

.. math::

    decay\_steps &= decay\_steps * math.ceil(\frac{global\_step}{decay\_steps})  \\
    decayed\_learning\_rate &= (learning\_rate-end\_learning\_rate)*(1-\frac{global\_step}{decay\_steps})^{power}+end\_learning\_rate

若cycle为False，则计算公式为：

.. math::

    global\_step &= min(global\_step, decay\_steps) \\
    decayed\_learning\_rate &= (learning\_rate-end\_learning\_rate)*(1-\frac{global\_step}{decay\_steps})^{power}+end\_learning\_rate

式中，

- :math:`decayed\_learning\_rate` ： 衰减后的学习率。
式子中各参数详细介绍请看参数说明。

参数
:::::::::
    - **learning_rate** (Tensor|float32) - 初始学习率。如果类型为Tensor，则为shape为[1]的Tensor，数据类型为float32或float64；也可以是python的float类型。
    - **gamma** （float）：衰减率。
    - **end_lr** (float，可选) - 最小的最终学习率。默认值为0.0001。
    - **power** (float，可选) - 多项式的幂。默认值为1.0。
    - **cycle** (bool，可选) - 学习率下降后是否重新上升。若为True，则学习率衰减到最低学习率值时，会出现上升。若为False，则学习率曲线则单调递减。默认值为False。
    - **last_epoch** （int，可选）: 上一轮的轮数，重启训练时设置为上一轮的epoch数。默认值为 -1，则为初始学习率。
    - **verbose** （bool，可选）：如果是 `True` ，则在每一轮更新时在标准输出 `stdout` 输出一条信息。

返回
:::::::::
返回计算PolynomialLR的可调用对象。


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

