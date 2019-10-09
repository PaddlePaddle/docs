.. _cn_api_fluid_layers_py_func:

py_func
-------------------------------

.. py:function:: paddle.fluid.layers.py_func(func, x, out, backward_func=None, skip_vars_in_backward_input=None)

PaddlePaddle Fluid通过该接口在Python端注册OP。所注册的Python OP的前向函数是 ``func``, 反向函数是 ``backward_func`` 。 Paddle将在前向部分调用 ``func`` ，并在反向部分调用 ``backward_func`` （如果 ``backward_func`` 不是None)。 ``x`` 为 ``func`` 的输入，必须为LoDTensor类型； ``out``  为 ``func`` 的输出， 既可以是LoDTensor类型, 也可以是NumPy数组。

反向函数 ``backward_func`` 的输入依次为：前向输入 ``x`` 、前向输出 ``out`` 、 ``out`` 的梯度。 如果 ``out`` 的某些变量没有梯度，则 ``backward_func`` 的相关输入变量为None。如果 ``x`` 的某些变量没有梯度，则用户应在 ``backward_func`` 中主动返回None。 

在调用该接口之前，还应正确设置 ``out`` 的数据类型和形状，而 ``out`` 和 ``x`` 对应梯度的数据类型和形状将自动推断而出。

此功能还可用于调试正在运行的网络，可以通过添加没有输出的 ``py_func`` 运算，并在 ``func`` 中打印输入 ``x`` 。

参数:
    - **func** （callable） - 所注册的Python OP的前向函数，运行网络时，将根据该函数与前向输入 ``x`` ，计算前向输出 ``out`` 。
    - **x** (Variable) -  前向函数 ``func`` 的输入，可以为 Variable | tuple[Variable] | list[Variale]， 其中 Variable 为LoDTensor类型。
    - **out** (Variable) -  前向函数 ``func`` 输出，可以为 Variable | tuple[Variable] | list[Variale]，其中 Variable 既可以为LoDTensor类型，也可以为NumPy数组。由于Paddle无法自动推断 ``out`` 的形状和数据类型，必须应事先创建 ``out`` 。
    - **backward_func** (callable，可选) - 所注册的Python OP的反向函数。默认值为None，意味着没有反向计算。若不为None，则会在运行网络反向时调用 ``backward_func`` 计算 ``x`` 的梯度。 
    - **skip_vars_in_backward_input** (Variable，可选) -  ``backward_func`` 的输入中不需要的变量，可以是 单个Variable | list[Variable] | tuple[Variable]。 这些变量必须是 ``x`` 和 ``out`` 中的一个。默认值为None，意味着没有变量需要从 ``x`` 和 ``out`` 中去除。若不为None，则这些变量将不是 ``backward_func`` 的输入。该参数仅在 ``backward_func`` 不为None时有用。

返回: 前向函数的输出 ``out``

返回类型: Variable | list[Variable] | tuple[Variable]

**代码示例**:

..  code-block:: python

    import paddle.fluid as fluid
    import six

    def create_tmp_var(name, dtype, shape):
        return fluid.default_main_program().current_block().create_var(
            name=name, dtype=dtype, shape=shape)

    # Paddle C++ op提供的tanh激活函数
    # 此处仅采用tanh作为示例展示py_func的使用方法
    def tanh(x):
        return np.tanh(x)

    # 跳过前向输入x
    def tanh_grad(y, dy):
        return np.array(dy) * (1 - np.square(np.array(y)))

    def debug_func(x):
        print(x)

    def simple_net(img, label):
        hidden = img
        for idx in six.moves.range(4):
            hidden = fluid.layers.fc(hidden, size=200)
            new_hidden = create_tmp_var(name='hidden_{}'.format(idx),
                dtype=hidden.dtype, shape=hidden.shape)

            # 用户自定义的前向反向计算
            hidden = fluid.layers.py_func(func=tanh, x=hidden,
                out=new_hidden, backward_func=tanh_grad,
                skip_vars_in_backward_input=hidden)

            # 用户自定义的调试层，可以打印出变量细则
            fluid.layers.py_func(func=debug_func, x=hidden, out=None)

        prediction = fluid.layers.fc(hidden, size=10, act='softmax')
        loss = fluid.layers.cross_entropy(input=prediction, label=label)
        return fluid.layers.mean(loss)






