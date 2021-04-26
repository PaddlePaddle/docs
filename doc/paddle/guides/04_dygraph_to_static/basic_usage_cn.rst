基本用法
==============

PaddlePaddle主要的动转静方式是基于源代码级别转换的ProgramTranslator。其基本原理是通过分析Python代码来将动态图代码转写为静态图代码，并在底层自动帮用户使用静态图执行器运行。这种转换方式使得用户可以灵活使用Python语法及其控制流来构建神经网络模型。除此之外，PaddlePaddle另外提供一种基于trace的动转静接口TracedLayer。若遇到ProgramTranslator不支持但是可以用TracedLayer运行的情况，可以作为备选方案。

基于源代码转写的ProgramTranslator
-----------------------------------

源代码转写的ProgramTranslator进行动态图转静态图，其基本原理是通过分析Python代码来将动态图代码转写为静态图代码，并在底层自动帮用户使用执行器运行。其基本使用方法十分简便，只需要在要转化的函数（该函数也可以是用户自定义动态图Layer的forward函数）前添加一个装饰器 ``@paddle.jit.to_static`` ，一个转化例子如下，可以直接运行被装饰函数得到结果：

.. code-block:: python

    import paddle
    import numpy as np

    @paddle.jit.to_static
    def func(input_var):
        # if判断与输入input_var的shape有关
        if input_var.shape[0] > 1:
            out = paddle.cast(input_var, "float64")
        else:
            out = paddle.cast(input_var, "int64")
        return out

    in_np = np.array([-2]).astype('int')
    input_var = paddle.to_tensor(in_np)
    func(input_var)


若要存储转化后的静态图模型，可以调用 ``paddle.jit.save`` ，我们定义一个简单全连接网络SimpleFcLayer，需要在下面SimpleFcLayer的forward函数添加装饰器：

.. code-block:: python

    import numpy as np
    import paddle

    class SimpleFcLayer(paddle.nn.Layer):
        def __init__(self, batch_size, feature_size, fc_size):
            super(SimpleFcLayer, self).__init__()
            self._linear = paddle.nn.Linear(feature_size, fc_size)
            self._offset = paddle.to_tensor(
                np.random.random((batch_size, fc_size)).astype('float32'))

        @paddle.jit.to_static
        def forward(self, x):
            fc = self._linear(x)
            return fc + self._offset


存储该模型可以使用 ``paddle.jit.save`` 接口：

.. code-block:: python

    import paddle

    fc_layer = SimpleFcLayer(3, 4, 2)
    in_np = np.random.random([3, 4]).astype('float32')
    input_var = paddle.to_tensor(in_np)
    out = fc_layer(input_var)

    paddle.jit.save(fc_layer, "./fc_layer_dy2stat", input_spec=[input_var])


基于trace的TracedLayer
------------------------

trace是指在模型运行时记录下其运行过哪些算子。TracedLayer就是基于这种技术，在一次执行动态图的过程中，记录所有运行的算子，并构建和保存静态图模型。一个使用例子如下：

我们还是定义一个简单的全连接网络作为例子，注意这里不需要像ProgramTranslator在forward函数添加装饰器：

.. code-block:: python

    import numpy as np
    import paddle

    class SimpleFcLayer(paddle.nn.Layer):
        def __init__(self, batch_size, feature_size, fc_size):
            super(SimpleFcLayer, self).__init__()
            self._linear = paddle.nn.Linear(feature_size, fc_size)
            self._offset = paddle.to_tensor(
                np.random.random((batch_size, fc_size)).astype('float32'))

        def forward(self, x):
            fc = self._linear(x)
            return fc + self._offset


接下来是TracedLayer如何存储模型：

.. code-block:: python

    import paddle
    from paddle.jit import TracedLayer

    fc_layer = SimpleFcLayer(3, 4, 2)
    in_np = np.random.random([3, 4]).astype('float32')
    # 将numpy的ndarray类型的数据转换为Tensor类型
    input_var = paddle.to_tensor(in_np)
    # 通过 TracerLayer.trace 接口将命令式模型转换为声明式模型
    out_dygraph, static_layer = TracedLayer.trace(fc_layer, inputs=[input_var])
    save_dirname = './saved_infer_model'
    # 将转换后的模型保存
    static_layer.save_inference_model(save_dirname, feed=[0], fetch=[0])


载入的模型可以使用静态图方式运行

.. code-block:: python

    paddle.enable_static()
    place = paddle.CPUPlace()
    exe = paddle.Executor(place)
    program, feed_vars, fetch_vars = paddle.static.load_inference_model(save_dirname, exe)
    fetch, = exe.run(program, feed={feed_vars[0]: in_np}, fetch_list=fetch_vars)


但是也正如我们阐述的原理，trace只是记录了一次执行涉及的算子。若在用户的模型代码中，包含了依赖数据条件（包括输入的值或者shape）的控制流分支，即根据数据条件触发运行不同的算子，则TracedLayer无法正常工作。比如下面：

.. code-block:: python

    import paddle

    def func(input_var)
        # if判断与输入input_var的shape有关
        if input_var.shape[0] > 1:
            return paddle.cast(input_var, "float64")
        else:
            return paddle.cast(input_var, "int64")

    in_np = np.array([-2]).astype('int')
    input_var = paddle.to_tensor(in_np)
    out = func(input_var)


如果对上述样例中的 ``func`` 使用 ``TracedLayer.trace(func, inputs=[input_var])`` ，由于trace只能记录if-else其中跑的一次算子，模型就无法按用户想要的根据input_var的形状进行if-else控制流保存。类似的控制流还有while/for循环的情况。

比较ProgramTranslator和TracedLayer
------------------------------------
基于源代码转换的ProgramTranslator对比基于trace的TracedLayer，前者能够处理依赖数据条件的控制流分支。因此我们更推荐用户使用ProgramTranslator，如果遇到问题再以TracedLayer作为备选方案。

