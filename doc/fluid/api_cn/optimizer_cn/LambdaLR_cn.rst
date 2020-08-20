.. _cn_api_optimizer_LambdaLR

LambdaLR
-----------------------------------

.. py:class:: paddle.optimizer.lr_scheduler.LambdaLR(learning_rate, lr_lambda, last_epoch=-1, verbose=False)

该接口提供 ``lambda`` 函数设置学习率的功能。 ``lr_lambda`` 为一个 ``lambda`` 函数，其通过 ``epoch`` 计算出一个因子，该因子会乘以初始学习率。。

衰减过程可以参考以下代码：

.. code-block:: python

    learning_rate = 0.5        # init learning_rate
    lr_lambda = lambda epoch: 0.95 ** epoch
    learning_rate = 0.5        # epoch 0
    learning_rate = 0.475      # epoch 1
    learning_rate = 0.45125    # epoch 2


参数
:::::::::
    - **learning_rate** （float|int）：初始学习率，可以是Python的float或int。
    - **lr_lambda** （float|int）：lr_lambda 为一个lambda函数，其通过 epoch 计算出一个因子，该因子会乘以初始学习率。
    - **last_epoch** （int，optional）: 上一轮的轮数，重启训练时设置为上一轮的epoch数。默认值为 -1，则为初始学习率 。
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
