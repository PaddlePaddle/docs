.. _cn_api_paddle_optimizer_lr_PolynomialDecay:

PolynomialDecay
-------------------------------

.. py:class:: paddle.optimizer.lr.PolynomialDecay(learning_rate, decay_steps, end_lr=0.0001, power=1.0, cycle=False, last_epoch=-1, verbose=False)


该接口提供学习率按多项式衰减的策略。通过多项式衰减函数，使得学习率值逐步从初始的 ``learning_rate``，衰减到 ``end_lr`` 。

若 cycle 为 True，则计算公式为：

.. math::

    decay\_steps & = decay\_steps * math.ceil(\frac{epoch}{decay\_steps})

    new\_learning\_rate & = (learning\_rate-end\_lr)*(1-\frac{epoch}{decay\_steps})^{power}+end\_lr

若 cycle 为 False，则计算公式为：

.. math::

    epoch & = min(epoch, decay\_steps)

    new\_learning\_rate & = (learning\_rate-end\_lr)*(1-\frac{epoch}{decay\_steps})^{power}+end\_lr


参数
::::::::::::

    - **learning_rate** (float) - 初始学习率，数据类型为 Python float。
    - **decay_steps** (int) - 进行衰减的步长，这个决定了衰减周期。
    - **end_lr** (float，可选）- 最小的最终学习率。默认值为 0.0001。
    - **power** (float，可选) - 多项式的幂，power 应该大于 0.0，才能使学习率衰减。默认值为 1.0。
    - **cycle** (bool，可选) - 学习率下降后是否重新上升。若为 True，则学习率衰减到最低学习率值时，会重新上升。若为 False，则学习率单调递减。默认值为 False。
    - **last_epoch** (int，可选) - 上一轮的轮数，重启训练时设置为上一轮的 epoch 数。默认值为 -1，则为初始学习率。
    - **verbose** (bool，可选) - 如果是 `True`，则在每一轮更新时在标准输出 `stdout` 输出一条信息。默认值为 ``False`` 。

返回
::::::::::::
用于调整学习率的 ``PolynomialDecay`` 实例对象。


代码示例
::::::::::::

COPY-FROM: paddle.optimizer.lr.PolynomialDecay

方法
::::::::::::
step(epoch=None)
'''''''''

step 函数需要在优化器的 `optimizer.step()` 函数之后调用，调用之后将会根据 epoch 数来更新学习率，更新之后的学习率将会在优化器下一轮更新参数时使用。

**参数**

  - **epoch** (int，可选) - 指定具体的 epoch 数。默认值 None，此时将会从-1 自动累加 ``epoch`` 数。

**返回**

无。

**代码示例**

参照上述示例代码。
