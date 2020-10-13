.. _cn_api_fluid_backward_append_backward:

append_backward
-------------------------------


.. py:function:: paddle.static.append_backward(loss, parameter_list=None, no_grad_set=None, callbacks=None)




该接口将向主程序（``main_program``）添加反向部分 。

完整的神经网络训练由前向和反向传播两部分组成。但是当我们配置网络时，我们只需要指定其前向部分。
该接口使用链式法则，能够根据前向部分自动生成反向部分。

在大多数情况下，用户无需手动调用此接口，它将由优化器（``Optimizer``）的 ``minimize`` 函数自动调用。

参数：
    - **loss** (Tensor) - 表示网络损失的 Tensor 。
    - **parameter_list** （list [Tensor|str]，可选）- 指定优化器需要更新的参数或参数名称列表。如果为 ``None`` ，则将更新所有参数。默认值为 ``None``。
    - **no_grad_set** （set [Tensor|str]，可选）-  在 `block0` ( :ref:`api_guide_Block` ) 中要忽略梯度的 Tensor 的名字的集合。所有的 :ref:`api_guide_Block` 中带有 ``stop_gradient = True`` 的所有 Tensor 的名字都会被自动添加到此集合中。如果该参数不为 ``None``，则会将该参数集合的内容添加到默认的集合中。默认值为 ``None``。
    - **callbacks** （list [callable object]，可选）- 回调函数列表。用于在反向传播构建中执行一些自定义作业。每次将新的梯度 OP 添加到程序中时，将调用其中的所有可调用对象。可调用对象必须有两个输入参数： :ref:`api_guide_Block` 和 ``context`` 。 :ref:`api_guide_Block` 是将被添加到新梯度算子的块。 ``context`` 是一个映射，其键是梯度 Tensor 名，值是对应的原始 Tensor 。除此之外， ``context`` 还有另一个特殊的键值对：键是字符串 ``__ current_op_desc__`` ，值是刚刚触发可调用对象的梯度 OP 的 ``op_desc`` 。默认值为 ``None``。

返回：   参数及其梯度 Tensor 的元组的列表。元组的第一个值为参数，第二个值为该参数的梯度 Tensor 。

返回类型：       list[(Tensor , Tensor)]

抛出：     
    - ``AssertionError`` - 如果 loss 不是 Tensor 的实例。

**示例代码**

.. code-block:: python

        import paddle
        import paddle.nn.functional as F

        paddle.enable_static()

        x = paddle.static.data(name='x', shape=[None, 13], dtype='int64')
        y = paddle.static.data(name='y', shape=[None, 1], dtype='float32')
        x_emb = paddle.static.nn.embedding(x, size=[100, 256])
        y_predict = paddle.static.nn.fc(x=x_emb, size=1, activation=None, name='my_fc')
        loss = F.square_error_cost(input=y_predict, label=y)
        avg_loss = paddle.mean(loss)

        # Get all weights in main_program, not include bias.
        all_weights = [param for param in paddle.static.default_main_program().block(0).all_parameters() if 'w_' in param.name]
        all_weights_name = [w.name for w in all_weights]

        # return all param_grads needed to be updated if parameter_list set default None.
        p_g_list1 = paddle.static.append_backward(loss=avg_loss)
        # output: [(embedding_0.w_0, embedding_0.w_0@GRAD), (my_fc.w_0, my_fc.w_0@GRAD), (my_fc.b_0, my_fc.b_0@GRAD)]

        # return the param_grads corresponding to parameter_list that can be list of param (Tensor).
        p_g_list2 = paddle.static.append_backward(loss=avg_loss, parameter_list=all_weights)
        # output: [(embedding_0.w_0, embedding_0.w_0@GRAD), (my_fc.w_0, my_fc.w_0@GRAD)]

        # parameter_list can be list of param.name (str).
        p_g_list3 = paddle.static.append_backward(loss=avg_loss, parameter_list=all_weights_name)
        # output: [(embedding_0.w_0, embedding_0.w_0@GRAD), (my_fc.w_0, my_fc.w_0@GRAD)]

        # no_grad_set can be set of Tensors that means grad will be cut off from these Tensors.
        p_g_list4 = paddle.static.append_backward(loss=avg_loss, no_grad_set=set([x_emb]))
        # output: [(my_fc.w_0, my_fc.w_0@GRAD), (my_fc.b_0, my_fc.b_0@GRAD)]

        # no_grad_set can be set of Tensor.name when the Tensors is created inside layers and can't be specified explicitly.
        p_g_list5 = paddle.static.append_backward(loss=avg_loss, no_grad_set=set(['my_fc.b_0']))
        # output: [(embedding_0.w_0, embedding_0.w_0@GRAD), (my_fc.w_0, my_fc.w_0@GRAD)]

        # return [] because all param_grads are filtered by no_grad_set.
        p_g_list6 = paddle.static.append_backward(loss=avg_loss, parameter_list=all_weights, no_grad_set=set(all_weights))



