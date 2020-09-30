动态图转静态图
================

动态图有诸多优点，包括易用的接口，python风格的编程体验，友好的debug交互机制等。在动态图模式下，代码是按照我们编写的顺序依次执行。这种机制更符合Python程序员的习惯，可以很方便地将大脑中的想法快速地转化为实际代码，也更容易调试。但在性能方面，Python执行开销较大，与C++有一定差距。因此在工业界的许多部署场景中（如大型推荐系统、移动端）都倾向于直接使用C++来提速。

相比动态图，静态图在部署方面更具有性能的优势。静态图程序在编译执行时，先搭建模型的神经网络结构，然后再对神经网络执行计算操作。预先搭建好的神经网络可以脱离Python依赖，在C++端被重新解析执行，而且拥有整体网络结构也能进行一些网络结构的优化。

动态图代码更易编写和debug，但在部署性能上，静态图更具优势。因此我们新增了动态图转静态图的功能，支持用户依然使用动态图编写组网代码。PaddlePaddle会对用户代码进行分析，自动转换为静态图网络结构，兼顾了动态图易用性和静态图部署性能两方面优势。

基本使用方法
--------------

PaddlePaddle提供了两种动态图转静态图的方式，基于动态图trace的TracedLayer与基于源代码级别转换的ProgramTranslator。

1. 基于trace的TracedLayer：

trace是指在模型运行时记录下其运行过哪些算子。TracedLayer就是基于这种技术，在一次执行动态图的过程中，记录所有运行的算子，并构建和保存静态图模型。一个使用例子如下：

我们先定义一个简单的Fully Connected网络：

.. code-block:: python

    import numpy as np
    import paddle

    class SimpleFcLayer(paddle.nn.Layer):
        def __init__(self, feature_size, batch_size, fc_size):
            super(SimpleFCLayer, self).__init__()
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

    paddle.disable_static()

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

    place = paddle.CPUPlace()
    exe = paddle.Executor(place)
    program, feed_vars, fetch_vars = paddle.static.load_inference_model(save_dirname, exe)
    fetch, = exe.run(program, feed={feed_vars[0]: in_np}, fetch_list=fetch_vars)


但是也正如我们阐述的原理，trace只是记录了一次执行涉及算子，若在用户的模型代码中，包含了依赖数据条件（包括输入的值或者shape）的控制流分支，即根据数据条件触发运行不同的算子，则TracedLayer无法正常工作。比如下面


.. code-block:: python

    import paddle

    def func(input_var)
        # if判断与输入input_var的shape有关
        if input_var.shape[0] > 1:
            return paddle.cast(input_var, "float64")
        else:
            return paddle.cast(input_var, "int64")

    paddle.disable_static()
    in_np = np.array([-2]).astype('int')
    input_var = paddle.to_tensor(in_np)
    out = func(input_var)


上例如果在使用TracedLayer.trace(func, inputs=[input_var])，由于trace只能记录if-else其中跑的一次算子，模型就无法按用户想要的根据input_var的形状进行if-else控制流保存。类似的控制流还有while/for循环的情况

2. 基于源代码转写的ProgramTranslator

对于依赖数据的控制流，我们使用基于源代码转写的ProgramTranslator来进行动态图转静态图。其基本原理是通过分析Python代码来将动态图代码转写为静态图代码，并在底层自动帮用户使用执行器运行。其基本使用方法十分简便，只需要在要转化的函数（该函数也可以是用户自定义动态图Layer的forward函数）前添加一个装饰器 ``@paddle.jit.to_static`` ，上面的例子转化如下，并且可以依旧使用该函数运行得到结果：

.. code-block:: python

    import paddle

    @paddle.jit.to_static
    def func(input_var)
        # if判断与输入input_var的shape有关
        if input_var.shape[0] > 1:
            out = paddle.cast(input_var, "float64")
        else:
            out = paddle.cast(input_var, "int64")

    paddle.disable_static()
    in_np = np.array([-2]).astype('int')
    input_var = paddle.to_tensor(in_np)
    func(input_var)


若要存储转化后的静态图模型，可以调用 ``paddle.jit.save`` ，我们再以SimpleFcLayer为例，需要在SimpleFcLayer的forward函数添加装饰器：

.. code-block:: python

    import numpy as np
    import paddle

    class SimpleFcLayer(paddle.nn.Layer):
        def __init__(self, feature_size, batch_size, fc_size):
            super(SimpleFCLayer, self).__init__()
            self._linear = paddle.nn.Linear(feature_size, fc_size)
            self._offset = paddle.to_tensor(
                np.random.random((batch_size, fc_size)).astype('float32'))

        @paddle.jit.to_static
        def forward(self, x):
            fc = self._linear(x)
            return fc + self._offset


存储该模型可以使用paddle.jit.save接口：

.. code-block:: python

    import paddle

    paddle.disable_static()

    fc_layer = SimpleFcLayer(3, 4, 2)
    in_np = np.random.random([3, 4]).astype('float32')
    input_var = paddle.to_tensor(in_np)
    out = fc_layer(input_var)

    paddle.jit.save(fc_layer, "./fc_layer_dy2stat", input_spec=[input_var])

内部架构原理
--------------

TracedLayer的原理就是trace，相对简单，因此我们在这里不展开描述。本节将主要阐述ProgramTranslator基于源代码将动态图代码转化为静态图代码。


转化过程发生在用户开始调用被装饰的函数，转换过程在装饰器中实现。我们将内部涉及的过程分为以下几步：

1. 函数与缓存

动态图转静态图的主体是函数（Function）。对于函数内包含的PaddlePaddle接口，如果是仅计算相关算子代码语句，那么因为PaddlePaddle动态图和静态图接口一致，我们不需要额外转换这些代码为静态图代码。但是对于动态图，此类代码接口是直接运行计算和返回结果，而对于静态图此类代码接口其实是组网。那么如果被转化的函数被调用多次，动态图转静态图后会多次组网添加对应算子，这显然会导致问题。为了解决这个问题以及为了加速动转静转化过程，我们维护了被装饰器装饰的函数（Function）与其输入形状（shape），数据类型（dtype）映射到被转化后组网的Program的缓存（Cache）。当要被转化的函数命中缓存，我们直接用对应存储的Program运行静态图得到结果，否则我们才进行语句转化，并且转化成功后的Program存储进缓存。

2. 动态图源码转AST（抽象语法树）

动态图转静态图的最核心部分类似一个编译器，解析动态图代码语句为AST，再对应AST进行改写，最后反转回成静态图代码。从函数转化为代码字符串可以使用Python的inspect.getsource。从字符串Python提供了自带的 `ast <https://docs.python.org/3/library/ast.html>`_ 库来解析字符串为AST，但是由于Python2，Python3的语法略有不同，为了避免我们需要额外处理这些Python2，Python3的不同情况，我们使用了统一Python2，Python3的开源AST处理 `gast库 <https://github.com/serge-sans-paille/gast>`_ 。这些接口使得函数转化为AST没有本质上的困难。

3. AST改写和静态图源码转换

这部分为动转静最核心的部分，我们对支持的各种语法进行ast转写。其中最重要的Python控制流，if-else，while，for循环被分别分析转化为PaddlePaddle静态图接口cond，while_loop等接口实现。我们对想转化的每一种主要语法创建一个Transformer（这里的Transformer是Python ast转写的概念，而不是自然语言处理NLP领域的Transformer），每个Transformer扫一遍AST并进行对应的改写。最后被转化完成的AST我们使用gast提供的接口转回成源码。

4. 静态图源码作为动态图一部分运行的技术

为了动静转化更加易用和被转化的代码能在动态图中复用，我们在拥有源码后运行生成Program，并将这个Program作为一个大op，包装成动态图的一个op，这样既能把用户的代码转为静态图提速或者保存部署，另一方面如果用户想在Python层使用生成的静态图代码作为动态图的一部分继续训练或者别的动态图运算也是可以直接使用。

5. 易用性与Debug功能在动转静过程的实现

正如AST转写类似编译器，而一般编译器都会提供debug断点，报错，输出一些中间代码等功能。我们在进行动转静时，万一用户的动态图代码出错，或者用户想断点调试，或者用户想看看被转化后的静态图代码是否符合其预期，我们也希望能够像编译器一样提供这些易用性功能，使得动转静兼顾性能和部署同时还具有易用性。我们这里将列出这些功能的实现方式

A. 报错对应到动态图代码行。由于被转化后的静态图代码和原动态图代码不同，Python运行出错时会报静态图的错误，因此我们在每一次AST转写时添加AST节点对应的原动态图代码行等信息，在Python报错栈中将静态图的报错转化成对应的动态图源码报错

B. 设置断点功能。我们保留了被转化后代码的中的pdb.set_trace(), 用户可以使用这种方式进行断点调试

C. 查看最后转化的静态图代码。我们输出为一个StaticLayer class，这个StaticLayer可以直接被调用，但是也存储转化后的代码，可以调用StaticLayer.code来获得转化后的代码。

D. 输出中间转化状态代码，甚至不同语法Transformer转化的代码，比如经过for循环转化后代码是什么样的。我们开放接口设定了log level来让用户可以打印中间状态转化的代码。


