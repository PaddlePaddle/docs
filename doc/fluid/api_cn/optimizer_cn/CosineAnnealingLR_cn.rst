.. _cn_api_optimizer_CosineAnnealingLR

LambdaLR
-----------------------------------

.. py:class:: paddle.optimizer.lr_scheduler.CosineAnnealingLR(learning_rate, T_max, eta_min=0, last_epoch=-1, verbose=False) 

该接口使用 ``cosine annealing`` 方式来动态调整学习率，根据下面的公式动态调整学习率。

.. math::
        \begin{aligned}
            \eta_t & = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1
            + \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right),
            & T_{cur} \neq (2k+1)T_{max}; \\
            \eta_{t+1} & = \eta_{t} + \frac{1}{2}(\eta_{max} - \eta_{min})
            \left(1 - \cos\left(\frac{1}{T_{max}}\pi\right)\right),
            & T_{cur} = (2k+1)T_{max}.
        \end{aligned}


:math:`\eta_{max}` 的初始值为 ``learning_rate``, :math:`T_{cur}` 是SGDR（重启训练SGD）训练过程中的当前训练轮数。SGDR的训练方法可以参考文档 `SGDR: Stochastic Gradient Descent with Warm Restarts <https://arxiv.org/abs/1608.03983>`_.
这里只是实现了 ``cosine annealing`` 动态学习率，热启训练部分没有实现。 


参数
:::::::::
    - **learning_rate** （float|int）：初始学习率，可以是Python的float或int。
    - **T_max** （float|int）：训练的上限轮数。
    - **eta_min** （float|int, optional）：学习率的下限，即公式中的 :math:`\eta_{min}` 。 
    - **last_epoch** （int，optional）: 上一轮的轮数，重启训练时设置为上一轮的epoch数。默认值为 -1，则为初始学习率。
    - **verbose** （bool）：如果是 `True` ，则在每一轮更新时在标准输出 `stdout` 输出一条信息。

返回
:::::::::
    返回计算LambdaLR的可调用对象。

代码示例
:::::::::

.. code-block:: python


.. py:method:: step(epoch)

通过当前的 ``step`` 函数调整学习率，调整后的学习率将会在下个step生效。

参数：
  - **epoch** (int|float，可选) - 类型：int或float。指定当前的step数。默认：无，此时将会自动累计step数。

返回：
    无

**代码示例**:

    参照上述示例代码。
