..  _api_guide_parameter_en:

##################
Model Parameters
##################

.. note::
  The `paddle.fluid.*` APIs are deprecated. Please use the latest Paddle API versions instead.

Model parameters are usually weights and bias in the model. In Paddle's dynamic graph mode, they correspond to the :code:`EagerParamBase` class. Model parameters are learnable variables that have gradients and can be optimized. In Paddle, custom parameters can be created using :ref:`api_paddle_create_parameter` .

You can configure properties related to model parameters using :ref:`api_paddle_ParamAttr` . The configurable options include:

- Initialization method
- Regularization
- Model averaging
- Clipping

Initialization Method
=====================

Paddle initializes a single parameter by setting attributes of :code:`initializer` in :code:`ParamAttr` .Example:

  .. code-block:: python

      import paddle

      param_attrs = paddle.ParamAttr(name="fc_weight",
                                initializer=paddle.nn.initializer.Constant(5.0))
      fc_layer = paddle.nn.Linear(64, 10, weight_attr=param_attrs)

The following is the initialization method supported by Paddle:

1. Constant
------------

The constant initialization method sets parameters to a fixed value, such as initializing biases to 0.

- Parameter: `value` specifies the initial value (default is 0.0).

API reference: :ref:`api_paddle_nn_initializer_Constant`

2. Normal
----------

The random normal distribution method generates values based on a normal (Gaussian) distribution, suitable for initializing most neural network parameters.

- Parameters: `mean` (default 0.0) and `std` (default 1.0) define the mean and standard deviation.

API reference: :ref:`api_paddle_nn_initializer_Normal`

3. Uniform
-----------

The random uniform distribution method samples values evenly within a specified range [low, high].

- Parameters: `low` (default -1.0) and `high` (default 1.0) define the range.

API reference: :ref:`api_paddle_nn_initializer_Uniform`

4. XavierUniform
-----------------

The Xavier uniform distribution method, proposed by Xavier Glorot and Yoshua Bengio in the paper **Understanding the difficulty of training deep feedforward neural networks**, initializes parameters based on a uniform distribution.

- Range is determined by `fan_in` (input dimension), `fan_out` (output dimension), and `gain` (scaling factor).

API reference: :ref:`api_paddle_nn_initializer_XavierUniform`

5. XavierNormal
----------------

The Xavier normal distribution method, proposed in the paper **Understanding the difficulty of training deep feedforward neural networks**, initializes parameters with a normal distribution.

- Mean is 0, and the standard deviation is determined by `fan_in`, `fan_out`, and `gain` .

API reference: :ref:`api_paddle_nn_initializer_XavierNormal`

6. KaimingUniform
------------------

The Kaiming uniform distribution method, proposed by Kaiming He et al. in the paper **Delving Deep into Rectifiers**, is designed for networks with specific activation functions.

- Range is determined by `fan_in`, `negative_slope` (default 0), and `nonlinearity` (default 'relu').

API reference: :ref:`api_paddle_nn_initializer_KaimingUniform`

7. KaimingNormal
-----------------

The Kaiming normal distribution method, proposed in the paper **Delving Deep into Rectifiers**, uses a normal distribution.

- Mean is 0, and the standard deviation is determined by `fan_in`, `negative_slope` (default 0), and `nonlinearity` (default 'relu').

API reference: :ref:`api_paddle_nn_initializer_KaimingNormal`

8. TruncatedNormal
-------------------

The truncated normal distribution method limits the generated values from a normal distribution to a specified range [a, b].

- Parameters: `mean` (default 0.0), `std` (default 1.0), and truncation bounds `a` and `b` (default -2.0 and 2.0).

API reference: :ref:`api_paddle_nn_initializer_TruncatedNormal`

Other Initialization Methods
----------------------------

Paddle also supports the following initialization methods:

- :ref:`api_paddle_nn_initializer_Assign`: Initialize directly using a NumPy array, Python list, or Tensor.
- :ref:`api_paddle_nn_initializer_Bilinear`: Used for upsampling in transposed convolutions to enlarge feature maps.
- :ref:`api_paddle_nn_initializer_Dirac`: Initializes convolution kernels with a Dirac delta function to preserve input characteristics.
- :ref:`api_paddle_nn_initializer_Orthogonal`: Generates an orthogonal matrix for initialization, ensuring (semi-)orthogonality.
- :ref:`api_paddle_nn_initializer_set_global_initializer`: Sets a global initialization method, effective only for code that follows it.

Regularization
=================

Paddle regularizes a single parameter by setting attributes of :code:`regularizer` in :code:`ParamAttr` .

  .. code-block:: python

      import paddle

      param_attrs = paddle.ParamAttr(name="fc_weight",
                                regularizer=paddle.regularizer.L1Decay(0.1))
      fc_layer = paddle.nn.Linear(64, 10, weight_attr=param_attrs)

The following is the regularization approach supported by Paddle:

- :ref:`api_paddle_regularizer_L1Decay`
- :ref:`api_paddle_regularizer_L2Decay`

Model Averaging
================

Paddle determines whether to average a single parameter by setting attributes of :code:`do_model_average` in :code:`ParamAttr` .

- Default value: `True` .

  .. code-block:: python

      import paddle

      param_attrs = paddle.ParamAttr(name="fc_weight",
                                do_model_average=True)
      fc_layer = paddle.nn.Linear(64, 10, weight_attr=param_attrs)

During mini-batch training, the model parameters are updated after each batch. Model averaging calculates the average of the parameters from the most recent k updates.

The averaged parameters are used only for testing and prediction, not for training.

API reference: :ref:`api_paddle_incubate_ModelAverage` (currently in incubation and may undergo changes).

Clipping
==========

.. note::
  The :code:`gradient_clip` attribute is deprecated. Use :code:`need_clip` to control whether gradient clipping is applied, and configure the clipping method when initializing the :code:`optimizer` .

Paddle determines whether gradient clipping is applied to a single parameter by setting attributes of :code:`need_clip` in :code:`ParamAttr` .

- Default value: `True` .

  .. code-block:: python

      import paddle

      param_attrs = paddle.ParamAttr(name="fc_weight",
                                need_clip=True)
      fc_layer = paddle.nn.Linear(64, 10, weight_attr=param_attrs)

The following is the clipping method supported by Paddle:

1. GradientClipByGlobalNorm
---------------------------

Limits the sum of the L2 norms of all Tensors in a Tensor list `t_list` to the :code:`clip_norm` range.

API reference: :ref:`api_paddle_nn_ClipGradByGlobalNorm`

2. GradientClipByNorm
---------------------

Limits the L2 norm of a multi-dimensional input Tensor `X` to the :code:`clip_norm` range.

API reference: :ref:`api_paddle_nn_ClipGradByNorm`

3. GradientClipByValue
----------------------

Limits the values of a multi-dimensional input Tensor `X` to the range [min, max].

API reference: :ref:`api_paddle_nn_ClipGradByValue`

For more details on gradient clipping methods, refer to: :ref:`Gradient clip methods in Paddle <en_gradient_clip>` .
