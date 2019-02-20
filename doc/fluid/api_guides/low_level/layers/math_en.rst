_api_guide_math_en:


Mathematical operation
###########################

Paddle provides a wealth of mathematical operations. The mathematical operations listed below are all elementwise operations on the target tensor. If the two inputs of the binary operations have different shapes, they will be processed first by :code:`broadcast`. Some mathematical operations also support mathematical operators, such as: :code:`+`, :code:`-`, :code:`*`, :code:`/`, etc. Math operators not only support tensors but also scalars.


Unary operation
==================

exp
------------------

Perform an :code:`exp` operation on each input :code:`Tensor` element.

API Reference:  :ref:`api_fluid_layers_exp`

tanh
------------------

For the input :code:`Tensor`, take the tanh value of each element.

API Reference:  :ref:`api_fluid_layers_tanh`

sqrt
------------------

For the input :code:`Tensor`, take the square root of each element.

API Reference:  :ref:`api_fluid_layers_sqrt`

abs
------------------

For the input :code:`Tensor`, take the elementwise absolute value.

API Reference:  :ref:`api_fluid_layers_abs`

ceil
------------------

Round up each input :code:`Tensor` element to the nearest greater integer.

API Reference:  :ref:`api_fluid_layers_ceil`

floor
------------------

Round down each input :code:`Tensor` element to the nearest less integer.

API Reference:  :ref:`api_fluid_layers_floor`

sin
------------------

For the input :code:`Tensor`, take the elementwise sin value.

API Reference:  :ref:`api_fluid_layers_sin`

cos
------------------

For input :code:`Tensor`, take the elementwise cosine value.

API Reference:  :ref:`api_fluid_layers_cos`

round
------------------

Rounding the input :code:`Tensor` in elementwise order.

API Reference:  :ref:`api_fluid_layers_round`

square
------------------

Square the input :code:`Tensor` in elementwise order.

API Reference:  :ref:`api_fluid_layers_square`

reciprocal
------------------

For the input :code:`Tensor`, take the reciprocal in elementwise order.

API Reference:  :ref:`api_fluid_layers_reciprocal`


reduce
------------------

For the input :code:`Tensor`, it performs reduce operations on the specified axes, including: min, max, sum, mean, product

API Reference:
:ref:`api_fluid_layers_reduce_min`
:ref:`api_fluid_layers_reduce_max`
:ref:`fluid_layers_reduce_sum`
:ref:`api_fluid_layers_reduce_mean`
:ref:`api_fluid_layers_reduce_prod`


Binary operation
==================

elementwise_add
------------------

Add two :code:`Tensor` in elementwise order, and the corresponding math operator is :code:`+` .

API Reference:  :ref:`api_fluid_layers_elementwise_add`

elementwise_sub
------------------

Sub two :code:`Tensor` in elementwise order, the corresponding math operator is :code:`-` .

API Reference:  :ref:`api_fluid_layers_elementwise_sub`

elementwise_mul
------------------

Multiply two :code:`Tensor` in elementwise order, and the corresponding math operator is :code:`*` .

API Reference:  :ref:`api_fluid_layers_elementwise_mul`

elementwise_div
------------------

Divide two :code:`Tensor` in elementwise order, and the corresponding math operator is :code:`/` or :code:`//` .

API Reference:  :ref:`api_fluid_layers_elementwise_div`


elementwise_pow
------------------

Do power operations on two :code:`Tensor` in elementwise order, and the corresponding math operator is :code:`**` .

API Reference:  :ref:`api_fluid_layers_elementwise_pow`

equal
------------------

Judge whether the two :code:`Tensor` elements are equal, and the corresponding math operator is :code:`==` .

API Reference:  :ref:`api_fluid_layers_equal`


less_than
------------------

Judge whether the two :code:`Tensor` elements satisfy the 'less than' relationship, and the corresponding math operator is :code:`<` .

API Reference:  :ref:`api_fluid_layers_less_than`



sum
------------------

Add two :code:`Tensor` in elementwise order.

API Reference:  :ref:`api_fluid_layers_sum`

elementwise_min
------------------

Perform :code:`min(x, y)` operations on two :code:`Tensor` in elementwise order .

API Reference:  :ref:`api_fluid_layers_elementwise_min`

elementwise_max
------------------

Perform :code:`max(x, y)` operations on two :code:`Tensor` in elementwise order .

API Reference:  :ref:`api_fluid_layers_elementwise_max`

matmul
------------------

Perform matrix multiplication operations on two :code:`Tensor`.

API Reference:  :ref:`api_fluid_layers_matmul`
