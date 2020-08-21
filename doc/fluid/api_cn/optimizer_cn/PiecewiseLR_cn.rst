.. _cn_api_fluid_optimizer_PiecewiseLR:

PiecewiseLR
-------------------------------


.. py:class:: paddle.optimizer.lr_scheduler.PiecewiseLR(boundaries, values, last_epoch=-1, verbose=False)


该接口提供对初始学习率进行分段(piecewise)常数衰减的功能。

分段常数衰减的过程举例描述如下。

.. code-block:: text

    例如，设定的boundaries列表为[10000, 20000]，候选学习率常量列表values为[1.0, 0.5, 0.1]，则：
    1、在当前训练步数global_step小于10000步，学习率值为1.0。
    2、在当前训练步数global_step大于或等于10000步，并且小于20000步时，学习率值为0.5。
    3、在当前训练步数global_step大于或等于20000步时，学习率值为0.1。


参数
:::::::::
    - **boundaries** (list) ：指定衰减的步数边界。列表的数据元素为Python int类型。
    - **values** (list) ：备选学习率列表。数据元素类型为Python float的列表。与边界值列表有对应的关系。
    - **last_epoch** （int，可选）：上一轮的轮数，重启训练时设置为上一轮的epoch数。默认值为 -1，则为初始学习率 。
    - **verbose** （bool，可选）：如果是 `True` ，则在每一轮更新时在标准输出 `stdout` 输出一条信息。

返回
:::::::::
返回计算PiecewiseLR的可调用对象。

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



