.. _user_guide_dy2sta_input_spec_cn:

Introduction of InputSpec
===========================


In PaddlePaddle(Referred to as "Paddle"), The dygraph model can be converted to static program by decorating the outermost forward function of Layer with ``paddle.jit.to_static`` . But actual Tensor data should be feeded into the model to ensure that the shape of each Tensor in the network is correctly deduced in transformation. This transformation process needs to explicitly execute the forward function, which increases the cost of the interface. Meanwhile, the way that need feed Tensor data fails to customize the shape of inputs, such as assigning some dimensions to None.

Therefore, Paddle provides the InputSpec interface to perform the transformation more easily, and supports to customize the signature of input Tensor, such as shape, name and so on.


1. InputSpec interface
-------------------------

1.1 Construct InputSpec object
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The InputSpec interface is under the ``paddle.static`` directory. It's used to describe the Tensor's signature information: shape, dtype, name. See example as follows:

.. code-block:: python

    from paddle.static import InputSpec

    x = InputSpec([None, 784], 'float32', 'x')
    label = InputSpec([None, 1], 'int64', 'label')

    print(x)      # InputSpec(shape=(-1, 784), dtype=VarType.FP32, name=x)
    print(label)  # InputSpec(shape=(-1, 1), dtype=VarType.INT64, name=label)


In InputSpec initialization, only ``shape`` is a required parameter. ``dtype`` and ``name`` can be default with values ``Float32`` and ``None`` respectively.



1.2 Constructed from Tensor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An InputSpec object can be created directly from a Tensor by using ``inputSpec.from_tensor`` method. It has same ``shape`` and ``dtype`` as the source Tensor. See example as follows:

.. code-block:: python

    import numpy as np
    import paddle
    from paddle.static import InputSpec

    x = paddle.to_tensor(np.ones([2, 2], np.float32))
    x_spec = InputSpec.from_tensor(x, name='x')
    print(x_spec)  # InputSpec(shape=(2, 2), dtype=VarType.FP32, name=x)


.. note::
    If a new name is not specified in ``from_tensor`` , the name from source Tensor is used by default.


1.3 Constructed from numpy.ndarray
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An InputSpec object can also be created directly from an Numpy.ndarray by using the ``inputSpec.from_numpy`` method. It has same ``shape`` and ``dtype`` as the source ndarray. See example as follows:

.. code-block:: python

    import numpy as np
    from paddle.static import InputSpec

    x = np.ones([2, 2], np.float32)
    x_spec = InputSpec.from_numpy(x, name='x')
    print(x_spec)  # InputSpec(shape=(2, 2), dtype=VarType.FP32, name=x)


.. note::
    If a new name is not specified in ``from_numpy`` , ``None`` is used by default.


2. Basic usage
------------------

Currently, the decorator ``paddle.jit.to_static`` support ``input_spec`` argument. It is used to specify signature information such as ``shape`` , ``dtype`` , ``name`` for each Tensor corresponding to argument from decorated function. Users do not have to feed actual data explicitly to trigger the deduction of the shape in the network. The ``input_spec`` argument specified in ``to_static`` will be analyzed to construct input placeholder of the network.

At the same time, the ``input_spec`` allow us to easily define input Tensor shape. For example, specifying shape as ``[None, 784]`` , where ``None`` represents a variable length dimension.

2.1 Decorator mode of to_static
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A simple example as follows:

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


In the above example, ``input_spec`` in  ``to_static`` decorator is a list of InputSpec objects. It is used to specify signature information corresponding x and y. After instantiating SimpleNet, ``paddle.jit.save`` can be directly called to save the static graph model without executing any other code.

.. note::
    1. Only InputSpec objects are supported in input_spec argument, and types such as int, float, etc. are not supported temporarily.
    2. If you specify the input_spec argument, you need to add the corresponding InputSpec object for all non-default parameters of the decorated function. As above sample, only specifying signature information x is not supported.
    3. If the decorated function includes non-tensor parameters and input_spec is specified, make sure that the non-tensor parameters of the function have default values, such as ``forward(self, x, use_bn=False)`` .


2.2 Call to_static directly
^^^^^^^^^^^^^^^^^^^^^^^^^^^

If we want to train model in dygraph mode and only expect to save the inference model after training with specified the signature information. We can call ``to_static`` function directly while saving the model. See example as follows:

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

In the above example,  ``to_static(net, input_spec=...)`` can be used to process the model after training.  Paddle will recursively convert forward function to get the complete static program according to ``input_spec`` information. Meanwhile, it includes the trained parameters.


2.3 Support list and dict derivation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the above two examples, the arguments of the decorated forward function correspond to the InputSpec one to one. But when the decorated function takes arguments with a list or dict type, ``input_spec`` needs to have the same nested structure as the arguments.

If a function takes an argument of type list, the element in the ``input_spec`` must also be an InputSpec list containing the same elements. A simple example as follows:

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


The length of ``input_spec`` is 1 corresponding to argument inputs in forward function. ``input_spec[0]`` contains two InputSpec objects corresponding to two Tensor signature information of inputs.

If a function takes an argument of type dict, the element in the ``input_spec`` must also be an InputSpec dict containing the same keys. A simple example as follows:

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


The length of ``input_spec`` is 2 corresponding to arguments x and bias_info in forward function. The last element of ``input_spec``  is a InputSpec dict with same key corresponding to signature information of bias_info.


2.4 Specify non-Tensor arguments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently, the ``input_spec`` from ``to_static`` decorator only receives objects with ``InputSpec`` type. When the decorated function contains some non-Tensor arguments, such as Int, String or other python types, we recommend to use kwargs with default values as argument, see use_act in followed example.

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


In above example, use_act is equal to True if step is an odd number, and False if step is an even number. We support non-tensor argument applied to different values during training after conversion. Moreover, the shared parameters of the model can be updated during the training with different values. The behavior is consistent with the dynamic graph.

The default value of the kwargs is primarily used for saving inference model. The inference model and network parameters will be exported based on input_spec and the default values of kwargs. Therefore, it is recommended to set the default value of the kwargs arguments for prediction.
