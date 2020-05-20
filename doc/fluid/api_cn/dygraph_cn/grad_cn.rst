.. _cn_api_fluid_dygraph_grad:

grad
-------------------------------

**注意：该API仅支持【动态图】模式**

.. py:method:: paddle.fluid.dygraph.grad(outputs, inputs, grad_outputs=None, retain_graph=None, create_graph=False, only_inputs=True, allow_unused=False, no_grad_vars=None, backward_strategy=None)

对于每个 `inputs` ，计算所有 `outputs` 相对于其的梯度和。

参数:
    - **outputs** (Variable|list(Variable)|tuple(Variable)) – 用于计算梯度的图的输出变量，或多个输出变量构成的list/tuple。
    - **inputs** (Variable|list(Variable)|tuple(Variable)) - 用于计算梯度的图的输入变量，或多个输入变量构成的list/tuple。该API的每个返回值对应每个 `inputs` 的梯度。
    - **grad_outputs** (Variable|list(Variable|None)|tuple(Variable|None), 可选) - `outputs` 变量梯度的初始值。若 `grad_outputs` 为None，则 `outputs` 梯度的初始值均为全1的Tensor。若 `grad_outputs` 不为None，它必须与 `outputs` 的长度相等，此时，若 `grad_outputs` 的第i个元素为None，则第i个 `outputs` 的梯度初始值为全1的Tensor；若 `grad_outputs` 的第i个元素为Variable，则第i个 `outputs` 的梯度初始值为 `grad_outputs` 的第i个元素。默认值为None。
    - **retain_graph** (bool, 可选) - 是否保留计算梯度的前向图。若值为True，则前向图会保留，用户可对同一张图求两次反向。若值为False，则前向图会释放。默认值为None，表示值与 `create_graph` 相等。
    - **create_graph** (bool, 可选) - 是否创建计算过程中的反向图。若值为True，则可支持计算高阶导数。若值为False，则计算过程中的反向图会释放。默认值为False。
    - **only_inputs** (bool, 可选) - 是否只计算 `inputs` 的梯度。若值为False，则图中所有叶节点变量的梯度均会计算，并进行累加。若值为True，则只会计算 `inputs` 的梯度。默认值为True。only_inputs=False功能正在开发中，目前尚不支持。
    - **allow_unused** (bool, 可选) - 决定当某些 `inputs` 变量不在计算图中时抛出错误还是返回None。若某些 `inputs` 变量不在计算图中（即它们的梯度为None），则当allowed_unused=False时会抛出错误，当allow_unused=True时会返回None作为这些变量的梯度。默认值为False。
    - **no_grad_vars** (Variable|list(Variable)|tuple(Variable)|set(Variable), 可选) - 指明不需要计算梯度的变量。默认值为None。
    - **backward_strategy** (BackwardStrategy, 可选) - 计算梯度的策略。详见 :ref:`cn_api_fluid_dygraph_BackwardStrategy` 。默认值为None。

返回: 变量构成的tuple，其长度等于 `inputs` 中的变量个数，且第i个返回的变量是所有 `outputs` 相对于第i个 `inputs` 的梯度之和。

返回类型: tuple

**示例代码 1**
  .. code-block:: python

    import paddle
    import paddle.fluid as fluid
    
    
    def test_dygraph_grad(create_graph):
        with paddle.imperative.guard():
            x = paddle.ones(shape=[1], dtype='float32', device=None, out=None)
            x.stop_gradient = False
            y = x * x
    
            # Since y = x * x, dx = 2 * x
            dx = fluid.dygraph.grad(outputs=[y], inputs=[x], create_graph=
                create_graph, retain_graph=True)[0]
            z = y + dx
    
            # If create_graph = False, the gradient of dx
            # would not be backpropagated. Therefore,
            # z = x * x + dx, and x.gradient() = 2 * x = 2.0
    
            # If create_graph = True, the gradient of dx
            # would be backpropagated. Therefore,
            # z = x * x + dx = x * x + 2 * x, and
            # x.gradient() = 2 * x + 2 = 4.0
    
            z.backward()
            return x.gradient()
    
    
    print(test_dygraph_grad(create_graph=False))
    print(test_dygraph_grad(create_graph=True))

**示例代码 2**
  .. code-block:: python

    import paddle
    import paddle.fluid as fluid
    
    
    def test_dygraph_grad(create_graph):
        with paddle.imperative.guard():
            x = paddle.ones(shape=[1], dtype='float32', device=None, out=None)
            x.stop_gradient = False
            y = x * x
    
            # Since y = x * x, dx = 2 * x
            dx = fluid.dygraph.grad(outputs=[y], inputs=[x], create_graph=
                create_graph, retain_graph=True)[0]
            z = y + dx
    
            # If create_graph = False, the gradient of dx
            # would not be backpropagated. Therefore,
            # z = x * x + dx, and x.gradient() = 2 * x = 2.0
    
            # If create_graph = True, the gradient of dx
            # would be backpropagated. Therefore,
            # z = x * x + dx = x * x + 2 * x, and
            # x.gradient() = 2 * x + 2 = 4.0
    
            z.backward()
            return x.gradient()
    
    
    print(test_dygraph_grad(create_graph=False))
    print(test_dygraph_grad(create_graph=True))

