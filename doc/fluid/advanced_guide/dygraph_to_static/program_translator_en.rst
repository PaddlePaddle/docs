Dygraph to Static Graph
=======================

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
    program, feed_vars, fetch_vars = paddle.static.load_inference_model(save_dirname, exe)
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

The basic idea of TracedLayer is tracing, it is relatively simple so we won't expend here. This section will talk about the source code transformation of ProgramTranslator.

The transformation is implemented in the decorator so transformation happens when user calls the decorated function, the procedure includes these steps:

1. Function and cache.

The entity for transforming dygraph to static graph is the decorated function. For the PaddlePaddle APIs in the function, since they are same code under dygraph mode and static mode, we don't have to transform those code. However, those APIs are computation in dygraph model while they are building network in static graph mode, if the transformed functions are called multiple times, those APIs will build network multiple times in static graph, which can cause problem. To solve it as well as speed up the transformation, we maintain a cache that maps from function, input shapes, input data types to the Program built by the transformed function. If the function hits cache, we run the stored Program in static graph mode to get result, else we do the code transformation on the function and store the transformed Program into the cache.

2. From dygraph source code to AST (Abstract Syntax Tree)

The core of transforming dygraph to static graph is similar to a compiler, we parse the dygraph code into AST, change AST, then turn it back into static graph code. We use Python ``inspect.getsource`` to get the source code string of the function. Python provides ``ast`` library to parse string code into AST, but Python2, Python3 have slight grammar difference. To avoid the work to handle different grammars, we used an open source AST library `gast <https://github.com/serge-sans-paille/gast>`_ that provides compatibility AST among various Python versions. There is no essential difficulty to turn function into AST with these library.

3. Transform AST and turn it to static graph code

This part is the key part in ProgramTranslator, we modify AST for supported grammars. Those important Python control flows, such as ``if-elif-else, while, for`` loop are converted to PaddlePaddle static graph API ``cond, while_loop`` and so on. We created a Transformer (AST-to-AST Transformer in Python, not the Transformer in Natural Language Process) to transform each grammar. Every Transformer scans AST and modify it. Lastly, we turn AST back to source code string by ``gast`` library.

4. Running static graph code as part of dygraph

In order to increase usability and re-use the transformed static graph code in dygraph, we wrap the generated Program as an dygraph op, the op can run the forward and backward computation of transformed Program. Then we can not only speed up dygraph code or save it for deployment, but also enable user to run part of their dygraph code in static graph mode so that they can continue training or other dygraph computation in their dygraph code.

5. Error handling and Debug

Compiler usually supports debug functionality like breakpoint, throwing exception, print some mid-level codes. ProgramTranslator is similar to a compiler, users may would like to set breakpoints for debugging, or see whether the transformed static graph code is expected. So we also implemented those error handling and debug functionality. Here we list those functions and their implementation.

A. Report errors/exceptions on dygraph code line. Because the transformed static graph code is different to original dygraph code, when Python executes the static graph code, the exceptions will be reported at static graph code. To locate the corresponding dygraph code, we attach some informations such as line number on AST nodes when we transform AST, then we can re-write the static graph exception to the corresponding dygraph code exception.

B. We support ``pdb.set_trace()`` when running ProgramTranslator, user can add this line to set breakpoints.

C. Check the transformed static graph code. Our transformed output is a Python class named ``StaticLayer``, this class can be called, but it also stores the transformed code string. Users could call ``StaticLayer.code`` to get the converted code.

D. Print mid-level transformed code, such as what's the code after transforming ``for`` loop. We provide APIs to set log level to let user check the mid-level code.


