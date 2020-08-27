.. _cn_api_fluid_backward_append_backward:

append_backward
-------------------------------


.. py:function:: paddle.fluid.backward.append_backward(loss, parameter_list=None, no_grad_set=None, callbacks=None)

:api_attr: 声明式编程模式（静态图)



该接口将向主程序（``main_program``）追加反向部分 。

完整的神经网络训练由前向和反向传播组成。但是当我们配置网络时，我们只需要指定其前向部分。
该接口使用链式法则，能够根据前向部分自动生成反向部分。

在大多数情况下，用户无需手动调用此接口，它将由优化器（``Optimizer``）的 ``minimize`` 函数自动调用。

参数：
    - **loss** ( :ref:`api_guide_Variable` ) - 网络的损失变量。
    - **parameter_list** （list [Variable|str]，可选）- 指定优化器需要更新的参数或参数名称列表。如果为 ``None`` ，则将更新所有参数。默认值为 ``None``。
    - **no_grad_set** （set [Variable|str]，可选）-  在 `block0` ( :ref:`api_guide_Block` ) 中要忽略梯度的 :ref:`api_guide_Variable` 的名字的集合。所有的 :ref:`api_guide_Block` 中带有 ``stop_gradient = True`` 的所有 :ref:`api_guide_Variable` 的名字都会被自动添加到此集合中。如果该参数不为 ``None``，则会将该参数集合的内容添加到默认的集合中。默认值为 ``None``。
    - **callbacks** （list [callable object]，可选）- 回调函数列表。用于在反向传播构建中执行一些自定义作业。每次将新的梯度OP添加到程序中时，将调用其中的所有可调用对象。可调用对象必须有两个输入参数： :ref:`api_guide_Block` 和 ``context`` 。 :ref:`api_guide_Block` 是将被添加到新梯度算子的块。 ``context`` 是一个映射，其键是梯度 :ref:`api_guide_Variable` 名，值是对应的原始 :ref:`api_guide_Variable` 。除此之外， ``context`` 还有另一个特殊的键值对：键是字符串 ``__ current_op_desc__`` ，值是刚刚触发可调用对象的梯度OP的 ``op_desc`` 。默认值为 ``None``。

返回：   参数及其梯度 :ref:`api_guide_Variable` 的元组的列表。元组的第一个值为参数，第二个值为该参数的梯度 :ref:`api_guide_Variable` 。

返回类型：       list[( :ref:`api_guide_Variable` , :ref:`api_guide_Variable` )]

抛出：     
    - ``AssertionError`` - 如果 loss 不是 :ref:`api_guide_Variable` 的实例。

**示例代码**

.. code-block:: python

        import paddle.fluid as fluid

        x = fluid.data(name='x', shape=[None, 13], dtype='int64')
        y = fluid.data(name='y', shape=[None, 1], dtype='float32')
        x_emb = fluid.embedding(x, size=[100, 256])
        y_predict = fluid.layers.fc(input=x_emb, size=1, act=None, name='my_fc')
        loss = fluid.layers.square_error_cost(input=y_predict, label=y)
        avg_loss = fluid.layers.mean(loss)

        # 获取main_program中所有weight参数, 不包括bias.
        all_weights = [param for param in fluid.default_main_program().block(0).all_parameters() if 'w_' in param.name]
        all_weights_name = [w.name for w in all_weights]

        # 若parameter_list为默认值(None), 则返回包含所有param_grad的list
        p_g_list1 = fluid.backward.append_backward(loss=avg_loss)
        # output: [(embedding_0.w_0, embedding_0.w_0@GRAD), (my_fc.w_0, my_fc.w_0@GRAD), (my_fc.b_0, my_fc.b_0@GRAD)]

        # 返回与传入parameter_list对应的param_grad的list, 传入的parameter_list可以是 param(Variable类型)的list
        p_g_list2 = fluid.backward.append_backward(loss=avg_loss, parameter_list=all_weights)
        # output: [(embedding_0.w_0, embedding_0.w_0@GRAD), (my_fc.w_0, my_fc.w_0@GRAD)]

        # 传入的parameter_list也可以是值为param.name(str类型)的list
        p_g_list3 = fluid.backward.append_backward(loss=avg_loss, parameter_list=all_weights_name)
        # output: [(embedding_0.w_0, embedding_0.w_0@GRAD), (my_fc.w_0, my_fc.w_0@GRAD)]

        # no_grad_set可以是set[Variables]类型，表示梯度将在这些Variables处截断
        p_g_list4 = fluid.backward.append_backward(loss=avg_loss, no_grad_set=set([x_emb]))
        # output: [(my_fc.w_0, my_fc.w_0@GRAD), (my_fc.b_0, my_fc.b_0@GRAD)]

        # no_grad_set也可以是set[Variable.names]类型。当参数Variable是在layers内部创建，且不方便显式地指定时，可以使用set[Variable.names]
        p_g_list5 = fluid.backward.append_backward(loss=avg_loss, no_grad_set=set(['my_fc.b_0']))
        # output: [(embedding_0.w_0, embedding_0.w_0@GRAD), (my_fc.w_0, my_fc.w_0@GRAD)]

        # 返回为[], 因为所有的param_grad均被传入的no_grad_set过滤掉了
        p_g_list6 = fluid.backward.append_backward(loss=avg_loss, parameter_list=all_weights, no_grad_set=set(all_weights))



