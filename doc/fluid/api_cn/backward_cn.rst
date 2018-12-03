
#################
fluid.backward
#################
.. _cn_api_fluid_backward_append_backward:

append_backward
>>>>>>>>>>>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.backward.append_backward(loss, parameter_list=None, no_grad_set=None, callbacks=None)

将向 ``main_program`` 追加 ``backward`` 。

完整的神经网络训练由前向和反向传播组成。但是当我们配置网络时，我们只需要指定其前向部分。通过该功能，根据前向部分自动生成反向部分。

在大多数情况下，用户无需手动调用此功能。它将由优化程序的最小化函数自动调用。

参数：
    - **loss** （Variable）- 网络的损失变量。
    - **parameter_list** （list [string] | None）- 优化器需要更新的参数名称。如果为None，则将更新所有参数。默认值：None。
    - **no_grad_set** （set | None）- ``block`` 0中变量的梯度应该被忽略。所有 ``block`` 中带有 ``step_gradient = True`` 的所有变量都将自动添加到此集合中。默认值：None。
    - **callbacks** （list [callable object] | None）- 回调用于在反向传播构建中执行一些自定义作业。每次将新的梯度运算符添加到程序中时，将调用其中的所有可调用对象。可调用对象必须有两个输入参数： ``block`` 和 ``context`` 。 ``block`` 是将被添加到新梯度算子的块。 ``context`` 是一个映射，其键是梯度变量名，值是对应的原始变量。除此之外， ``context`` 还有另一个特殊的键值对：键是字符串 ``__ current_op_desc__`` ，值是刚刚触发可调用对象的梯度运算符的 ``op_desc`` 。

返回：   成对参数及其相应的梯度。键是参数，值是梯度变量。

返回类型：   	list[(Variable,Variable)]

抛出：     
    - ``AssertionError`` - 如果loss不是Variable的实例。

**示例代码**

..  code-block:: python

        # 网络配置
        # ...
        avg_loss = fluid.layers.mean(loss)
        param_grad_list = fluid.backward.append_backward(loss=avg_loss)












































