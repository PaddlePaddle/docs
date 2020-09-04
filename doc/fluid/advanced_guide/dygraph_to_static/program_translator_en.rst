ProgramTranslator
=================

The imperative-style coding of PaddlePaddle takes advantage of flexibility, Pythonic coding, and easy-to-debug interface. In dygraph mode, code immediately executes kernels and gets numerical results, which allows users to enjoy traditional Pythonic code order. Therefore it is efficient to transform idea into real code and simple to debug. However, Python code is usually slower than C++ thus lots of industrial systems (such as large recommend system, mobile devices) prefer to deploy with C++ implementation.

Static graph is better at speed and portability. Static graph builds the network structure during compiling time and then does computation. The built network intermediate representation can be executed in C++ and gets rids of Python dependency.

While dygraph has usability and debug benefits and static graph yields performance and deployment advantage, we adds functionality to convert dygraph to static graph. Users use imperative mode to write dygraph code and PaddlePaddle will analyze the Python syntax and turn it into network structure of static graph mode. Our approach retains both the usability of dygraph and portability of static graph.

Basic Usage
--------------

PaddlePaddle has two ways to transform dygraph to static graph. TracedLayer extracts computation graph through tracing and ProgramTranslator gets computation graph through source code transformation.


1. TracedLayer：

Tracing means recording the operators when running a model. TracedLayer is based on this technique. It runs dygraph program once and records all operators, then constructs static graph model and saves it. Now take a glance at an usage example:

Define a simple fully connected network:

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

Save model by TracedLayer:

.. code-block:: python
    import paddle
    from paddle.jit import TracedLayer

    paddle.disable_static()

    fc_layer = SimpleFcLayer(3, 4, 2)
    in_np = np.random.random([3, 4]).astype('float32')
    # Turn numpy ndarray into Tensor
    input_var = paddle.to_tensor(in_np)
    # Transforming imperative mode into declarative mode by TracerLayer.trace
    out_dygraph, static_layer = TracedLayer.trace(fc_layer, inputs=[input_var])
    save_dirname = './saved_infer_model'
    # Save the transformed model
    static_layer.save_inference_model(save_dirname, feed=[0], fetch=[0])

Load model and run it in static graph mode:

.. code-block:: python

    place = paddle.CPUPlace()
    exe = paddle.Executor(place)
    program, feed_vars, fetch_vars = paddle.io.load_inference_model(save_dirname, exe)
    fetch, = exe.run(program, feed={feed_vars[0]: in_np}, fetch_list=fetch_vars)

However, as tracing only records operators once, if user's code contains Tensor-dependent (including Tensor value or Tensor shape) control flow, that is the Tensor can cause different operators being executed, then TracedLayer cannot handle this case. For instance:

.. code-block:: python

    import paddle

    def func(input_var)
        # if condition depends on the shape of input_var
        if input_var.shape[0] > 1:
            return paddle.cast(input_var, "float64")
        else:
            return paddle.cast(input_var, "int64")

    paddle.disable_static()
    in_np = np.array([-2]).astype('int')
    input_var = paddle.to_tensor(in_np)
    out = func(input_var)

If we apply TracedLayer.trace(func, inputs=[input_var]) on above example, tracing can take record of operators in only one branch of if-else, then the model can not be saved as what user orignally means. The similar situations applies to while/for loop.

2. ProgramTranslator

For the Tensor-dependent control flow, we use source-code-translate based ProgramTranslator to convert dygraph into static graph. The basic idea is analyzing Python source code and turning into static graph code, then run the static graph code using Executor. The basic usage of ProgramTranslator is simple, put a decorator ``@paddle.jit.to_static`` before the definition of the function to transform (the function can also be a method of a class, e.g., the ``forward`` function of user-defined imperative Layer). Above Tensor-dependent example can be transformed correctly by ProgramTranslator as below:

.. code-block:: python

    import paddle

    @paddle.jit.to_static
    def func(input_var)
        # if condition depends on the shape of input_var
        if input_var.shape[0] > 1:
            out = paddle.cast(input_var, "float64")
        else:
            out = paddle.cast(input_var, "int64")

    paddle.disable_static()
    in_np = np.array([-2]).astype('int')
    input_var = paddle.to_tensor(in_np)
    func(input_var)

To save the transformed model, we can call ``paddle.jit.save`` . Let's take ``SimpleFcLayer`` as an example again, we put decorator at the ``forward`` method of ``SimpleFcLayer`` :

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


Calling ``paddle.jit.save`` to save above model:

.. code-block:: python

    import paddle

    paddle.disable_static()

    fc_layer = SimpleFcLayer(3, 4, 2)
    in_np = np.random.random([3, 4]).astype('float32')
    input_var = paddle.to_tensor(in_np)
    out = fc_layer(input_var)

    paddle.jit.save(fc_layer, "./fc_layer_dy2stat")


Architecture
--------------

TracedLayer的原理就是trace，相对简单，因此我们在这里不展开描述。本节将主要阐述ProgramTranslator基于源代码将动态图代码转化为静态图代码。


转化过程发生在用户开始调用被装饰的函数，转换过程在装饰器中实现。我们将内部涉及的过程分为以下几步：

1. 函数与缓存

动态图转静态图的主体是函数（Function）。对于函数内包含的PaddlePaddle接口，如果是仅计算相关算子代码语句，那么因为PaddlePaddle动态图和静态图接口一致，我们不需要额外转换这些代码为静态图代码。但是对于动态图，此类代码接口是直接运行计算和返回结果，而对于静态图此类代码接口其实是组网。那么如果被转化的函数被调用多次，动态图转静态图后会多次组网添加对应算子，这显然会导致问题。为了解决这个问题以及为了加速动转静转化过程，我们维护了被装饰器装饰的函数（Function）与其输入形状（shape），数据类型（dtype）映射到被转化后组网的Program的缓存（Cache）。当要被转化的函数命中缓存，我们直接用对应存储的Program运行静态图得到结果，否则我们才进行语句转化，并且转化成功后的Program存储进缓存。

2. 动态图源码转AST（抽象语法树）

动态图转静态图的最核心部分类似一个编译器，解析动态图代码语句为AST，再对应AST进行改写，最后反转回成静态图代码。从函数转化为代码字符串可以使用Python的inspect.getsource。从字符串Python提供了自带的ast库来解析字符串为 `AST <https://docs.python.org/3/library/ast.html>`_ ，但是由于python2，python3的语法略有不同，为了避免我们需要额外处理这些python2，python3的不同情况，我们使用了统一python2，python3的开源AST处理 `gast库 <https://github.com/serge-sans-paille/gast>`_ 。这些接口使得函数转化为AST没有本质上的困难。

3. AST改写和静态图源码转换

这部分为动转静最核心的部分，我们对支持的各种语法进行ast转写。其中最重要的python控制流，if-else，while，for循环被分别分析转化为PaddlePaddle静态图接口cond，while_loop等接口实现。我们对想转化的每一种主要语法创建一个Transformer（这里的Transformer是python ast转写的概念，而不是自然语言处理NLP领域的Transformer），每个Transformer扫一遍AST并进行对应的改写。最后被转化完成的AST我们使用gast提供的接口转回成源码。

4. 静态图源码作为动态图一部分运行的技术

为了动静转化更加易用和被转化的代码能在动态图中复用，我们在拥有源码后运行生成Program，并将这个Program作为一个大op，包装成动态图的一个op，这样既能把用户的代码转为静态图提速或者保存部署，另一方面如果用户想在python层使用生成的静态图代码作为动态图的一部分继续训练或者别的动态图运算也是可以直接使用。

5. 易用性与Debug功能在动转静过程的实现

正如AST转写类似编译器，而一般编译器都会提供debug断点，报错，输出一些中间代码等功能。我们在进行动转静时，万一用户的动态图代码出错，或者用户想断点调试，或者用户想看看被转化后的静态图代码是否符合其预期，我们也希望能够像编译器一样提供这些易用性功能，使得动转静兼顾性能和部署同时还具有易用性。我们这里将列出这些功能的实现方式

A. 报错对应到动态图代码行。由于被转化后的静态图代码和原动态图代码不同，python运行出错时会报静态图的错误，因此我们在每一次AST转写时添加AST节点对应的原动态图代码行等信息，在python报错栈中将静态图的报错转化成对应的动态图源码报错

B. 设置断点功能。我们保留了被转化后代码的中的pdb.set_trace(), 用户可以使用这种方式进行断点调试

C. 查看最后转化的静态图代码。我们输出为一个StaticLayer class，这个StaticLayer可以直接被调用，但是也存储转化后的代码，可以调用StaticLayer.code来获得转化后的代码。

D. 输出中间转化状态代码，甚至不同语法Transformer转化的代码，比如经过for循环转化后代码是什么样的。我们开放接口设定了log level来让用户可以打印中间状态转化的代码。


