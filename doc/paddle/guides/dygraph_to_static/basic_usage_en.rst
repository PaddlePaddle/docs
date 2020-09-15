Basic Usage
=============

PaddlePaddle has two ways to transform dygraph to static graph. TracedLayer extracts computation graph through tracing and ProgramTranslator gets computation graph through source code transformation.


TracedLayer
-------------

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

ProgramTranslator
-------------------

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


