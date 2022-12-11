Gradient clip methods in Paddle
========================================

Deep neural network learns by gradient descent. With the increase of the number of layers in network, the problem of "gradient explosion" may become more obvious. For example, in gradient back propagation, if the partial derivative of the output relative to the input in each layer is greater than 1, the gradient will become larger and larger.

If "gradient explosion" occurs, the optimal solution may be skipped. So it is necessary to clip gradient to avoid "gradient explosion".

Paddle provides three methods of gradient clip:

1. Clip gradient by value
----------------------------------------

Limits the gradient to a range. If it is outside this range, gradients will be clipped to this range.

How to use it? You need to create an instance of class :ref:`cn_api_fluid_clip_ClipGradByValue` and pass it to the ``optimizer`` , which will clip the gradient before updating the parameter.

**a. Clip all gradients**

By default, Gradients of all parameters in SGD optimizer will be clipped:

.. code:: ipython3

    import paddle

    linear = paddle.nn.Linear(10, 10)
    clip = paddle.nn.ClipGradByValue(min=-1, max=1)
    sdg = paddle.optimizer.SGD(learning_rate=0.1, parameters=linear.parameters(), grad_clip=clip)

You can also clip gradients of a part of parameters as follow:

**b. Clip a part of gradients**

You can clip a part of gradients by setting `need_clip` of ref:`cn_api_fluid_ParamAttr` . `need_clip` is `True` by default, which represents that its gradient will be clipped. Otherwise, its gradient will not be clipped.

For example:
If only clip the gradient of `weight` in `linear`, you should set `bias_attr` as follow:

.. code:: ipython3

    linear = paddle.nn.Linear(10, 10，bias_attr=paddle.ParamAttr(need_clip=False))

2. Clip gradient by norm
----------------------------------------

Assuming that gradient is a N-D Tensor ``X`` , if L2 norm of ``X`` exceeds ``clip_norm`` , ``X`` will be clipped and new L2 norm is ``clip_norm`` .

How to use it? You need to create an instance of class :ref:`cn_api_fluid_clip_ClipGradByValue` and pass it to the ``optimizer`` , which will clip the gradient before updating the parameter.

The formula is as follow:

.. math::

  Out=
  \left\{
  \begin{aligned}
  &  X & & if (norm(X) \leq clip\_norm)\\
  &  \frac{clip\_norm∗X}{norm(X)} & & if (norm(X) > clip\_norm) \\
  \end{aligned}
  \right.


where, :math:`norm（X）` represent L2 norm of :math:`X` .

.. math::
  \\norm(X) = (\sum_{i=1}^{n}|x_i|^2)^{\frac{1}{2}}\\

**a. Clip all gradients**

By default, Gradients of all parameters in SGD optimizer will be clipped:

.. code:: ipython3

    linear = paddle.nn.Linear(10, 10)
    clip = paddle.nn.ClipGradByNorm(clip_norm=1.0)
    sdg = paddle.optimizer.SGD(learning_rate=0.1, parameters=linear.parameters(), grad_clip=clip)

You can also clip gradients of a part of parameters as follow:

**b. Clip a part of gradients**

You can clip a part of gradients by setting `need_clip` of ref:`cn_api_fluid_ParamAttr` . `need_clip` is `True` by default, which represents that its gradient will be clipped. Otherwise, its gradient will not be clipped.

For example:
If only clip the gradient of `bias` in `linear`, you should set `weight_attr` as follow:

.. code:: ipython3

    linear = paddle.nn.Linear(10, 10, weight_attr=paddle.ParamAttr(need_clip=False))

3. Clip gradient by global norm
----------------------------------------

Concat the gradient of all parameters to a vector, then calculate L2 norm this vector. If the L2 norm exceeds ``clip_norm`` , each tensor of this vector will be clipped and new L2 norm of this vector is ``clip_norm`` .

How to use it? You need to create an instance of class :ref:`cn_api_fluid_clip_ClipGradByGlobalNorm` and pass it to the ``optimizer`` , which will clip the gradient before updating the parameter.

The formula is as follow:

.. math::

  Out[i]=
  \left\{
  \begin{aligned}
  &  X[i] & & if (global\_norm \leq clip\_norm)\\
  &  \frac{clip\_norm∗X[i]}{global\_norm} & & if (global\_norm > clip\_norm) \\
  \end{aligned}
  \right.


where:

.. math::
            \\global\_norm=\sqrt{\sum_{i=0}^{n-1}(norm(X[i]))^2}\\


where, :math:`norm（X）` represents L2 norm of :math:`X` .

**a. Clip all gradients**

By default, Gradients of all parameters in SGD optimizer will be clipped:

.. code:: ipython3

    linear = paddle.nn.Linear(10, 10)
    clip = paddle.nn.ClipGradByGloabalNorm(clip_norm=1.0)
    sdg = paddle.optimizer.SGD(learning_rate=0.1, parameters=linear.parameters(), grad_clip=clip)

You can also clip gradients of a part of parameters as follow:

**b. Clip a part of gradients**

You can clip a part of gradients by setting `need_clip` of ref:`cn_api_fluid_ParamAttr` . `need_clip` is `True` by default, which represents that its gradient will be clipped. Otherwise, its gradient will not be clipped. Refer to the sample code above.
