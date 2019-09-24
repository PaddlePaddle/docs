.. _cn_api_fluid_backward_append_backward:

append_backward
-------------------------------

.. py:function:: paddle.fluid.backward.append_backward(loss, parameter_list=None, no_grad_set=None, callbacks=None)

该接口将向主程序（``main_program``）追加反向部分 。

完整的神经网络训练由前向和反向传播组成。但是当我们配置网络时，我们只需要指定其前向部分。
该接口使用链式法则，能够根据前向部分自动生成反向部分。

在大多数情况下，用户无需手动调用此接口，它将由优化器（``Optimizer``）的 ``minimize`` 函数自动调用。

参数：
    - **loss** ( :ref:`api_guide_Variable` ) - 网络的损失变量。
    - **parameter_list** （list [str]，可选）- 指定优化器需要更新的参数名称列表。如果为 ``None`` ，则将更新所有参数。缺省值为 ``None``。
    - **no_grad_set** （set，可选）-  无梯度的 :ref:`api_guide_Variable` 的集合。`block0` ( :ref:`api_guide_Block` ) 中 :ref:`api_guide_Variable` 的梯度应该被忽略。所有的 :ref:`api_guide_Block` 中带有 ``stop_gradient = True`` 的所有 :ref:`api_guide_Variable` 都会被自动添加到此集合中。缺省值为 ``None``。
    - **callbacks** （list [callable object]，可选）- 回调函数列表。用于在反向传播构建中执行一些自定义作业。每次将新的梯度OP添加到程序中时，将调用其中的所有可调用对象。可调用对象必须有两个输入参数： :ref:`api_guide_Block` 和 ``context`` 。 :ref:`api_guide_Block` 是将被添加到新梯度算子的块。 ``context`` 是一个映射，其键是梯度 :ref:`api_guide_Variable` 名，值是对应的原始 :ref:`api_guide_Variable` 。除此之外， ``context`` 还有另一个特殊的键值对：键是字符串 ``__ current_op_desc__`` ，值是刚刚触发可调用对象的梯度OP的 ``op_desc`` 。缺省值为 ``None``。

返回：   参数及其梯度 :ref:`api_guide_Variable` 的元组的列表。元组的第一个值为参数，第二个值为该参数的梯度 :ref:`api_guide_Variable` 。

返回类型：       list[( :ref:`api_guide_Variable` , :ref:`api_guide_Variable` )]

抛出：     
    - ``AssertionError`` - 如果 loss 不是 :ref:`api_guide_Variable` 的实例。

**示例代码**

.. code-block:: python

        import paddle.fluid as fluid
        x = fluid.layers.data(name='x', shape=[13], dtype='float32')
        y = fluid.layers.data(name='y', shape=[1], dtype='float32') 
        
        y_predict = fluid.layers.fc(input=x, size=1, act=None)
        loss = fluid.layers.square_error_cost(input=y_predict, label=y)
        
        avg_loss = fluid.layers.mean(loss)
        param_grad_list = fluid.backward.append_backward(loss=avg_loss)



