.. _cn_api_fluid_layers_py_func:

py_func
-------------------------------

.. py:function:: paddle.fluid.layers.py_func(func, x, out, backward_func=None, skip_vars_in_backward_input=None)

PyFunc运算。

用户可以使用 ``py_func`` 在Python端注册operator。 ``func`` 的输入 ``x`` 是LoDTensor，输出可以是numpy数组或LoDTensor。 Paddle将在前向部分调用注册的 ``func`` ，并在反向部分调用 ``backward_func`` （如果 ``backward_func`` 不是None）。

在调用此函数之前，应正确设置 ``out`` 的数据类型和形状。 但是，``out`` 和 ``x`` 对应梯度的数据类型和形状将自动推断而出。

``backward_func`` 的输入顺序为：前向输入x，前向输出 ``out`` 和反向输入 ``out`` 的梯度。 如果 ``out`` 的某些变量没有梯度，则输入张量在Python端将为None。

如果in的某些变量没有梯度，则用户应返回None。

此功能还可用于调试正在运行的网络，可以通过添加没有输出的py_func运算，并在func中打印输入x。

参数:
    - **func** （callable） - 前向Python函数。
    - **x** (Variable|list(Variable)|tuple(Variable)) -  func的输入。
    - **out** (Variable|list(Variable)|tuple(Variable)) -  func的输出。 Paddle无法自动推断out的形状和数据类型。 应事先创建 ``out`` 。
    - **backward_func** (callable|None) - 反向Python函数。 None意味着没有反向计算。 默认None。
    - **skip_vars_in_backward_input** (Variable|list(Variable)|tuple(Variable)) -  backward_func输入中不需要的变量。 这些变量必须是x和out中的一个。 如果设置，这些变量将不是backward_func的输入，仅在backward_func不是None时有用。 默认None。

返回: 传入的 ``out``

返回类型: out (Variable|list(Variable)|tuple(Variable))

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






