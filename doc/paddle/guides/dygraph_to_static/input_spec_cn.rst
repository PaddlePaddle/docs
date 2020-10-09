.. _user_guide_dy2sta_input_spec_cn:

InputSpec 功能介绍
=================


在PaddlePaddle（下文简称：Paddle）框架中，可以通过 ``paddle.jit.to_static`` 装饰普通函数或 Layer 的最外层 forward 函数，将动态图模型转换为静态图执行。但在动转静时，需要给模型传入 Tensor 数据并执行一次前向，以保证正确地推导出网络中各 Tensor 的 shape 。此转换流程需要显式地执行一次动态图函数，增加了接口使用的成本；同时，传入实际 Tensor 数据则无法定制化模型输入的shape，如指定某些维度为 None 。

因此，Paddle 提供了 InputSpec 接口，可以更加便捷地执行动转静功能，以及定制化输入 Tensor 的 shape 、name 等信息。


一、InputSpec 对象构造方法
-------------------------

1.1 直接构造 InputSpec 对象
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

InputSpec 接口在 ``paddle.static`` 目录下，用于描述一个 Tensor 的签名信息：shape、dtype、name。使用样例如下：

.. code-block:: python

    from paddle.static import InputSpec

    x = InputSpec([None, 784], 'float32', 'x')
    label = InputSpec([None, 1], 'int64', 'label')

    print(x)      # InputSpec(shape=(-1, 784), dtype=VarType.FP32, name=x)
    print(label)  # InputSpec(shape=(-1, 1), dtype=VarType.INT64, name=label)


InputSpec 初始化中的只有 ``shape`` 是必须参数， ``dtype`` 和 ``name`` 可以缺省，默认取值分别为 ``float32`` 和 ``None`` 。



1.2 根据 Tensor 构造 InputSpec 对象
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

可以借助 ``InputSpec.from_tensor`` 方法，从一个 Tensor 直接创建 InputSpec 对象，其拥有与源 Tensor 相同的 ``shape`` 和 ``dtype`` 。使用样例如下：

.. code-block:: python

    import numpy as np
    import paddle
    from paddle.static import InputSpec

    paddle.disable_static()

    x = paddle.to_tensor(np.ones([2, 2], np.float32))
    x_spec = InputSpec.from_tensor(x, name='x')
    print(x_spec)  # InputSpec(shape=(2, 2), dtype=VarType.FP32, name=x)


.. note::
    若未在 ``from_tensor`` 中指定新的name，则默认使用与源Tensor相同的name。


1.3 根据 numpy.ndarray 构造 InputSpec 对象
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

也可以借助 ``InputSpec.from_numpy`` 方法，从一个 Numpy.ndarray 直接创建 InputSpec 对象，其拥有与源 ndarray 相同的 ``shape`` 和 ``dtype`` 。使用样例如下：

.. code-block:: python

    import numpy as np
    from paddle.static import InputSpec

    x = np.ones([2, 2], np.float32)
    x_spec = InputSpec.from_numpy(x, name='x')
    print(x_spec)  # InputSpec(shape=(2, 2), dtype=VarType.FP32, name=x)


.. note::
    若未在 ``from_numpy`` 中指定新的 name，则默认使用 None 。


二、基本使用方法
------------------

动转静 ``paddle.jit.to_static`` 装饰器支持 ``input_spec`` 参数，用于指定被装饰函数每个 Tensor 类型输入参数的 ``shape`` 、 ``dtype`` 、 ``name`` 等签名信息。不必再显式地传入 Tensor 数据以触发网络层 shape 的推导。 Paddle 会解析 ``to_static`` 中指定的 ``input_spec`` 参数，构建网络的起始输入，进行后续的模型组网。

同时，借助 ``input_spec`` 参数，可以自定义输入 Tensor 的 shape ，比如指定 shape 为 ``[None, 784]`` ，其中 ``None`` 表示变长的维度。

2.1 to_static 装饰器模式
^^^^^^^^^^^^^^^^^^^^^^^^^^

如下是一个简单的使用样例：

.. code-block:: python

    import paddle
    from paddle.jit import to_static
    from paddle.static import InputSpec
    from paddle.fluid.dygraph import Layer

    class SimpleNet(Layer):
        def __init__(self):
            super(SimpleNet, self).__init__()
            self.linear = paddle.nn.Linear(10, 3)

        @to_static(input_spec=[InputSpec(shape=[None, 10], name='x'), InputSpec(shape=[3], name='y')])
        def forward(self, x, y):
            out = self.linear(x)
            out = out + y
            return out

    net = SimpleNet()

    # save static model for inference directly
    paddle.jit.save(net, './simple_net')


在上述的样例中， ``to_static`` 装饰器中的 ``input_spec`` 为一个 InputSpec 对象组成的列表，用于依次指定参数 x 和 y 对应的 Tensor 签名信息。在实例化 SimpleNet 后，可以直接调用 ``paddle.jit.save`` 保存静态图模型，不需要执行任何其他的代码。

.. note::
    1. input_spec 参数中只支持 InputSpec 对象，暂不支持如 int 、 float 等类型。
    2. 若指定 input_spec 参数，则需为被装饰函数的所有必选参数都添加对应的 InputSpec 对象，如上述样例中，不支持仅指定 x 的签名信息。
    3. 若被装饰函数中包括非 Tensor 参数，且指定了 input_spec ，请确保函数的非 Tensor 参数都有默认值，如 ``forward(self, x, use_bn=False)``


2.2 to_static函数调用
^^^^^^^^^^^^^^^^^^^^

若期望在动态图下训练模型，在训练完成后保存预测模型，并指定预测时需要的签名信息，则可以选择在保存模型时，直接调用 ``to_static`` 函数。使用样例如下：

.. code-block:: python

    class SimpleNet(Layer):
        def __init__(self):
            super(SimpleNet, self).__init__()
            self.linear = paddle.nn.Linear(10, 3)

        def forward(self, x, y):
            out = self.linear(x)
            out = out + y
            return out

    net = SimpleNet()

    # train process (Pseudo code)
    for epoch_id in range(10):
        train_step(net, train_reader)
        
    net = to_static(net, input_spec=[InputSpec(shape=[None, 10], name='x'), InputSpec(shape=[3], name='y')])

    # save static model for inference directly
    paddle.jit.save(net, './simple_net')


如上述样例代码中，在完成训练后，可以借助 ``to_static(net, input_spec=...)`` 形式对模型实例进行处理。Paddle 会根据 input_spec 信息对 forward 函数进行递归的动转静，得到完整的静态图，且包括当前训练好的参数数据。


2.3 支持 list 和 dict 推导
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

上述两个样例中，被装饰的 forward 函数的参数均为 Tensor 。这种情况下，参数个数必须与 InputSpec 个数相同。但当被装饰的函数参数为list或dict类型时，``input_spec`` 需要与函数参数保持相同的嵌套结构。

当函数的参数为 list 类型时，input_spec 列表中对应元素的位置，也必须是包含相同元素的 InputSpec 列表。使用样例如下：

.. code-block:: python

    class SimpleNet(Layer):
        def __init__(self):
            super(SimpleNet, self).__init__()
            self.linear = paddle.nn.Linear(10, 3)

        @to_static(input_spec=[[InputSpec(shape=[None, 10], name='x'), InputSpec(shape=[3], name='y')]])
        def forward(self, inputs):
            x, y = inputs[0], inputs[1]
            out = self.linear(x)
            out = out + y
            return out


其中 ``input_spec`` 参数是长度为 1 的 list ，对应 forward 函数的 inputs 参数。 ``input_spec[0]`` 包含了两个 InputSpec 对象，对应于参数 inputs 的两个 Tensor 签名信息。

当函数的参数为dict时， ``input_spec`` 列表中对应元素的位置，也必须是包含相同键（key）的 InputSpec 列表。使用样例如下：

.. code-block:: python

    class SimpleNet(Layer):
        def __init__(self):
            super(SimpleNet, self).__init__()
            self.linear = paddle.nn.Linear(10, 3)

        @to_static(input_spec=[InputSpec(shape=[None, 10], name='x'), {'x': InputSpec(shape=[3], name='bias')}])
        def forward(self, x, bias_info):
            x_bias = bias_info['x']
            out = self.linear(x)
            out = out + x_bias
            return out


其中 ``input_spec`` 参数是长度为 2 的 list ，对应 forward 函数的 x 和 bias_info 两个参数。 ``input_spec`` 的最后一个元素是包含键名为 x 的 InputSpec 对象的 dict ，对应参数 bias_info 的 Tensor 签名信息。


2.4 指定非Tensor参数类型
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

目前，``to_static`` 装饰器中的 ``input_spec`` 参数仅接收 ``InputSpec`` 类型对象。若被装饰函数的参数列表除了 Tensor 类型，还包含其他如 Int、 String 等非 Tensor 类型时，推荐在函数中使用 kwargs 形式定义非 Tensor 参数，如下述样例中的 use_act 参数。

.. code-block:: python

    class SimpleNet(Layer):
        def __init__(self, ):
            super(SimpleNet, self).__init__()
            self.linear = paddle.nn.Linear(10, 3)
            self.relu = paddle.nn.ReLU()

        @to_static(input_spec=[InputSpec(shape=[None, 10], name='x')])
        def forward(self, x, use_act=False):
            out = self.linear(x)
            if use_act:
                out = self.relu(out)
            return out

    net = SimpleNet()
    adam = paddle.optimizer.Adam(parameters=net.parameters())

    # train model
    batch_num = 10
    for step in range(batch_num):
        x = paddle.rand([4, 10], 'float32')
        use_act = (step%2 == 0)
        out = net(x, use_act)
        loss = paddle.mean(out)
        loss.backward()
        adam.minimize(loss)
        net.clear_gradients()

    # save inference model with use_act=False
    paddle.jit.save(net, model_path='./simple_net')


在上述样例中，step 为奇数时，use_act 取值为 False ； step 为偶数时， use_act 取值为 True 。动转静支持非 Tensor 参数在训练时取不同的值，且保证了取值不同的训练过程都可以更新模型的网络参数，行为与动态图一致。

kwargs 参数的默认值主要用于保存推理模型。在借助 ``paddle.jit.save`` 保存预测模型时，动转静会根据 input_spec 和 kwargs 的默认值保存推理模型和网络参数。因此建议将 kwargs 参数默认值设置为预测时的取值。

更多关于动转静 ``to_static`` 搭配 ``paddle.jit.save/load`` 的使用方式，可以参考 :ref:`user_guide_model_save_load` 。