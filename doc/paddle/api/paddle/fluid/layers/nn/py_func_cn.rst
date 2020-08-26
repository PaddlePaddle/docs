.. _cn_api_fluid_layers_py_func:

py_func
-------------------------------


.. py:function:: paddle.fluid.layers.py_func(func, x, out, backward_func=None, skip_vars_in_backward_input=None)

:api_attr: 声明式编程模式（静态图)



PaddlePaddle Fluid通过py_func在Python端注册OP。py_func的设计原理在于Paddle中的LodTensor与numpy数组可以方便的互相转换，从而可使用Python中的numpy API来自定义一个Python OP。

该自定义的Python OP的前向函数是 ``func``, 反向函数是 ``backward_func`` 。 Paddle将在前向部分调用 ``func`` ，并在反向部分调用 ``backward_func`` （如果 ``backward_func`` 不是None)。 ``x`` 为 ``func`` 的输入，必须为LoDTensor类型； ``out``  为 ``func`` 的输出， 既可以是LoDTensor类型, 也可以是numpy数组。

反向函数 ``backward_func`` 的输入依次为：前向输入 ``x`` 、前向输出 ``out`` 、 ``out`` 的梯度。 如果 ``out`` 的某些变量没有梯度，则 ``backward_func`` 的相关输入变量为None。如果 ``x`` 的某些变量没有梯度，则用户应在 ``backward_func`` 中主动返回None。 

在调用该接口之前，还应正确设置 ``out`` 的数据类型和形状，而 ``out`` 和 ``x`` 对应梯度的数据类型和形状将自动推断而出。

此功能还可用于调试正在运行的网络，可以通过添加没有输出的 ``py_func`` 运算，并在 ``func`` 中打印输入 ``x`` 。

参数:
    - **func** （callable） - 所注册的Python OP的前向函数，运行网络时，将根据该函数与前向输入 ``x`` ，计算前向输出 ``out`` 。 在 ``func`` 建议先主动将LoDTensor转换为numpy数组，方便灵活的使用numpy相关的操作，如果未转换成numpy，则可能某些操作无法兼容。
    - **x** (Variable|tuple(Variable)|list[Variale]) -  前向函数 ``func`` 的输入，多个LoDTensor以tuple(Variable)或list[Variale]的形式传入，其中Variable为LoDTensor或Tenosr。
    - **out** (Variable|tuple(Variable)|list[Variale]) -  前向函数 ``func`` 的输出，可以为Variable|tuple(Variable)|list[Variale]，其中Variable既可以为LoDTensor或Tensor，也可以为numpy数组。由于Paddle无法自动推断 ``out`` 的形状和数据类型，必须应事先创建 ``out`` 。
    - **backward_func** (callable，可选) - 所注册的Python OP的反向函数。默认值为None，意味着没有反向计算。若不为None，则会在运行网络反向时调用 ``backward_func`` 计算 ``x`` 的梯度。 
    - **skip_vars_in_backward_input** (Variable，可选) -  ``backward_func`` 的输入中不需要的变量，可以是Variable|tuple(Variable)|list[Variale]。 这些变量必须是 ``x`` 和 ``out`` 中的一个。默认值为None，意味着没有变量需要从 ``x`` 和 ``out`` 中去除。若不为None，则这些变量将不是 ``backward_func`` 的输入。该参数仅在 ``backward_func`` 不为None时有用。

返回: 前向函数的输出 ``out``

返回类型: Variable|tuple(Variable)|list[Variable]

**示例代码1**:

..  code-block:: python

    import paddle.fluid as fluid
    import six

    # 自定义的前向函数，可直接输入LoDTenosor
    def tanh(x):
        return np.tanh(x)

    # 在反向函数中跳过前向输入x，返回x的梯度。
    # 必须使用np.array主动将LodTensor转换为numpy，否则"+/-"等操作无法使用
    def tanh_grad(y, dy):
        return np.array(dy) * (1 - np.square(np.array(y)))

    # 自定义的前向函数，可用于调试正在运行的网络（打印值）
    def debug_func(x):
        print(x)
    
    def create_tmp_var(name, dtype, shape):
        return fluid.default_main_program().current_block().create_var(
            name=name, dtype=dtype, shape=shape)

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

            # 用户自定义的调试函数，打印出输入的LodTensor
            fluid.layers.py_func(func=debug_func, x=hidden, out=None)

        prediction = fluid.layers.fc(hidden, size=10, act='softmax')
        loss = fluid.layers.cross_entropy(input=prediction, label=label)
        return fluid.layers.mean(loss)

**示例代码2**:

..  code-block:: python
    
    # 该示例展示了如何将LoDTensor转化为numpy数组，并利用numpy API来自定义一个OP
    import paddle.fluid as fluid
    import numpy as np

    def element_wise_add(x, y): 
        # 必须先手动将LodTensor转换为numpy数组，否则无法支持numpy的shape操作
        x = np.array(x)    
        y = np.array(y)

        if x.shape != y.shape:
            raise AssertionError("the shape of inputs must be the same!")

        result = np.zeros(x.shape, dtype='int32')
        for i in range(len(x)):
            for j in range(len(x[0])):
                result[i][j] = x[i][j] + y[i][j]

        return result

    def create_tmp_var(name, dtype, shape):
        return fluid.default_main_program().current_block().create_var(
                    name=name, dtype=dtype, shape=shape)

    def py_func_demo():
        start_program = fluid.default_startup_program()
        main_program = fluid.default_main_program()

        # 创建前向函数的输入变量
        x = fluid.data(name='x', shape=[2,3], dtype='int32')
        y = fluid.data(name='y', shape=[2,3], dtype='int32')
        
        # 创建前向函数的输出变量，必须指明变量名称name/数据类型dtype/维度shape
        output = create_tmp_var('output','int32', [3,1])

        # 输入多个LodTensor以list[Variable]或tuple(Variable)形式
        fluid.layers.py_func(func=element_wise_add, x=[x,y], out=output)

        exe=fluid.Executor(fluid.CPUPlace())
        exe.run(start_program)

        # 给program喂入numpy数组
        input1 = np.random.randint(1, 10, size=[2,3], dtype='int32')
        input2 = np.random.randint(1, 10, size=[2,3], dtype='int32')
        out = exe.run(main_program, 
                    feed={'x':input1, 'y':input2},
                    fetch_list=[output.name])
        print("{0} + {1} = {2}".format(input1, input2, out))

    py_func_demo()

    # 参考输出：
    # [[5, 9, 9]   + [[7, 8, 4]  =  [array([[12, 17, 13]
    #  [7, 5, 2]]     [1, 3, 3]]            [8, 8, 5]], dtype=int32)]
