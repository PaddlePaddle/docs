InputSpec功能介绍
============


在PaddlePaddle（下文简称：Paddle）框架中，可以通过`paddle.jit.to_static`装饰最外层`forward`函数，将动态图模型转换为静态图执行。但在动转静时，需要给模型喂入Tensor数据并执行一次前向，以保证能够正确地推导出网络中张量的shape。此转换流程要求用户需要显式地执行一次动态图函数，增加了用户使用接口的成本；同时，喂入实际Tensor数据则无法定制化模型输入的shape，如指定某些维度为None。

因此，Paddle提供了InputSpec接口，支持用户更加简便地进行动转静，以及定制化输入Tensor的shape、name等信息。


一、InputSpec接口
------------------

1.1 构造InputSpec对象
^^^^^^^^^^^^^^^^^^^^^^

InputSpec接口暴露在`paddle.static`目录下，用于描述一个Tensor的签名信息：shape、dtype、name。使用样例如下：

.. code-block:: python

    from paddle.static import InputSpec

    x = InputSpec([None, 784], 'float32', 'x')
    label = InputSpec([None, 1], 'int64', 'label')

    print(x)      # InputSpec(shape=(-1, 784), dtype=VarType.FP32, name=x)
    print(label)  # InputSpec(shape=(-1, 1), dtype=VarType.INT64, name=label)


InputSpec初始化中的只有`shape`是必须参数，`dtype`和`name`可以缺省，默认取值分别为`float32`和`None`。



1.2 用Tensor构造
^^^^^^^^^^^^^^^^^^^^^^^^^^

可以借助`InputSpec.from_tensor`方法，从一个Tensor直接创建InputSpec对象，其拥有与源Tensor相同的`shape`和`dtype`。使用样例如下：

.. code-block:: python

    import numpy as np
    import paddle
    from paddle.static import InputSpec

    paddle.disable_static()

    x = paddle.to_tensor(np.ones([2, 2], np.float32))
    x_spec = InputSpec.from_tensor(x, name='x')
    print(x_spec)  # InputSpec(shape=(2, 2), dtype=VarType.FP32, name=x)


.. note::
    若未在`from_tensor`中指定新的`name`，则默认使用与源Tensor相同的`name`。


1.3 用numpy.ndarray构造
^^^^^^^^^^^^^^^^^^^^^^^^^^

也可以借助`InputSpec.from_numpy`方法，从一个Numpy.ndarray直接创建InputSpec对象，其拥有与源ndarray相同的`shape`和`dtype`。使用样例如下：
import numpy as np
from paddle.static import InputSpec

.. code-block:: python

    x = np.ones([2, 2], np.float32)
    x_spec = InputSpec.from_numpy(x, name='x')
    print(x_spec)  # InputSpec(shape=(2, 2), dtype=VarType.FP32, name=x)


.. note::
    若未在`from_numpy`中指定新的`name`，则默认使用None。


二、基本使用方法
------------------

在动转静`paddle.jit.to_static`装饰器中，支持`input_spec`参数，用于指定被装饰函数每个Tensor类型输入参数的`shape`、`dtype`、`name`等签名信息。用户不必再显式地喂入Tensor数据以触发网络层shape的推导。Paddle会解析用户在`to_static`中指定的`input_spec`参数，构建网络的起始输入，进行后续的模型组网。

同时，借助`input_spec`参数，可以友好地支持用户自定义输入Tensor的shape，比如指定shape为`[None, 784]`，其中`None`表示batch size的维度。

2.1 to_static装饰器模式
^^^^^^^^^^^^^^^^^^

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


    paddle.disable_static()

    net = SimpleNet()

    # save static model for inference directly
    paddle.jit.save(net, './simple_net')


在上述的样例中，`to_static`装饰器中的`input_spec`为一个InputSpec组成的列表，用于依次指定参数`x`和`y`对应的InputSpec签名信息。在实例化`SimpleNet`后，可以直接调用`paddle.jit.save`保存静态图模型，不要执行任何其他的代码。

.. note::
    1. input_spec参数中只支持InputSpec对象，暂不支持如int、float等类型。
    2. 若指定input_spec参数，则需为被装饰函数的所有非默认值参数都添加对应的InputSpec对象，如上述样例中不支持仅指定`x`的签名信息。
    3. 若被装饰函数中包括非Tensor参数，且指定了`input_spec`，请确保函数的非Tensor参数都有默认值，如`forward(self, x, use_bn=False)`


2.2 to_static函数调用
^^^^^^^^^^^^^^^^^^^^

若用户模型训练依旧使用原生动态图，只期望在训练完成后，保存预测模型，并指定预测时需要的签名信息。可以选择在保存模型时，直接调用`to_static`函数。使用样例如下：

.. code-block:: python

    class SimpleNet(Layer):
        def __init__(self):
            super(SimpleNet, self).__init__()
            self.linear = paddle.nn.Linear(10, 3)

        def forward(self, x, y):
            out = self.linear(x)
            out = out + y
            return out

    paddle.disable_static()
    net = SimpleNet()

    # train process
    for epoch_id in range(10):
        train_step(net, train_reader)
        
    net = to_static(net, input_spec=[InputSpec(shape=[None, 10], name='x'), InputSpec(shape=[3], name='y')])

    # save static model for inference directly
    paddle.jit.save(net, './simple_net')


如上述样例代码中，在完成训练后，可以借助`to_static(net, input_spec=...)`形式对模型实例进行处理。Paddle会根据`input_spec`信息对`forward`函数进行递归的动转静，得到完整的静态图，且包括当前训练好的参数数据。


2.3 支持list和dict推导
^^^^^^^^^^^^^^^^^^^^

上述两个样例中，被装饰的`forward`函数的参数与InputSpec都是一一对应。Paddle也支持被装饰的函数参数为list或dict类型。

当函数的参数为list类型时，`input_spec`列表中对应元素的位置，也必须是包含相同元素的InputSpec列表。使用样例如下：

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


其中`input_spec`参数是长度为1的list，对应`forward`函数的`inputs`参数。`input_spec[0]`包含了两个InputSpec对象，对应于参数`inputs`的两个Tensor签名信息。

当函数的参数为dict时，`input_spec`列表中对应元素的位置，也必须是包含相同键（key）的InputSpec列表。使用样例如下：

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


其中`input_spec`参数是长度为2的list，对应`forward`函数的`x`和`bias_info`两个参数。`input_spec`的最后一个元素是包含键名为`x`的InputSpec对象的dict，对应参数`bias_info`的Tensor签名信息。
