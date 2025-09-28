.. _cn_api_paddle_static_append_backward:

append_backward
-------------------------------


.. py:function:: paddle.static.append_backward(loss, parameter_list=None, no_grad_set=None, callbacks=None)




将向主程序（``main_program``）添加反向部分。

完整的神经网络训练由前向和反向传播两部分组成。但是当我们配置网络时，我们只需要指定其前向部分。
该接口使用链式法则，能够根据前向部分自动生成反向部分。

在大多数情况下，用户无需手动调用此接口，它将由优化器（``Optimizer``）的 ``minimize`` 函数自动调用。

参数
::::::::::::

    - **loss** (Tensor) - 表示网络损失的 Tensor 。
    - **parameter_list** （list [Tensor|str]，可选）- 指定优化器需要更新的参数或参数名称列表。如果为 ``None``，则将更新所有参数。默认值为 ``None``。
    - **no_grad_set** （set [Tensor|str]，可选）-  在 `block0` ( :ref:`api_guide_Block` ) 中要忽略梯度的 Tensor 的名字的集合。所有的 :ref:`api_guide_Block` 中带有 ``stop_gradient = True`` 的所有 Tensor 的名字都会被自动添加到此集合中。如果该参数不为 ``None``，则会将该参数集合的内容添加到默认的集合中。默认值为 ``None``。
    - **callbacks** （list [callable object]，可选）- 回调函数列表。用于在反向传播构建中执行一些自定义作业。每次将新的梯度 OP 添加到程序中时，将调用其中的所有可调用对象。可调用对象必须有两个输入参数：:ref:`api_guide_Block` 和 ``context`` 。 :ref:`api_guide_Block` 是将被添加到新梯度算子的块。``context`` 是一个映射，其键是梯度 Tensor 名，值是对应的原始 Tensor。除此之外，``context`` 还有另一个特殊的键值对：键是字符串 ``__ current_op_desc__``，值是刚刚触发可调用对象的梯度 OP 的 ``op_desc``。默认值为 ``None``。

返回
::::::::::::
   list[(Tensor , Tensor)]，参数及其梯度 Tensor 的元组的列表。元组的第一个值为参数，第二个值为该参数的梯度 Tensor 。

代码示例
::::::::::::

COPY-FROM: paddle.static.append_backward
