.. _api_guide_math:


Mathematical operation
#########

Paddle provides a wealth of mathematical operations. The mathematical operations listed below are all elemental operations on the target tensor. Among them, if the two inputs of the binary operation have different shapes, they will be executed first :code:`broadcast`. Some mathematical operations also support mathematical operators, such as: :code:`+`, :code:`-`, :code:`*`, :code:`/`, etc. Math operators not only support tensors but also scalars.


Unary operation
==================

exp
------------------

Do an :code:`exp` operation on the input :code:`Tensor` element by element .

API Reference please refer to :ref:`cn_api_fluid_layers_exp`

tanh
------------------

For the input :code:`Tensor`, take the tangent by element.

API Reference please refer to :ref:`cn_api_fluid_layers_tanh`

sqrt
------------------

For the input :code:`Tensor`, take the square root of by element.

API Reference please refer to :ref:`cn_api_fluid_layers_sqrt`

abs
------------------

For the input :code:`Tensor`, take the absolute value by element.

API Reference please refer to :ref:`cn_api_fluid_layers_abs`

ceil
------------------

Round up the input :code:`Tensor` element by element.

API Reference please refer to :ref:`cn_api_fluid_layers_ceil`

floor
------------------

Round down the input :code:`Tensor` element by element.

API Reference See :ref:`cn_api_fluid_layers_floor`

sin
------------------

For the input :code:`Tensor`, take the sine by element.

API Reference please refer to :ref:`cn_api_fluid_layers_sin`

cos
------------------

For input: code:`Tensor`, take the cosine by element.

API Reference please refer to :ref:`cn_api_fluid_layers_cos`

round
------------------

Rounding up the input :code:`Tensor` by element.

API Reference please refer to :ref:`cn_api_fluid_layers_round`

square
------------------

Square the input :code:`Tensor` by element.

API Reference please refer to :ref:`cn_api_fluid_layers_square`

reciprocal
------------------

For the input :code:`Tensor`, take the reciprocal by element.

API Reference please refer to :ref:`cn_api_fluid_layers_reciprocal`


reduce
------------------

For the input :code:`Tensor` to do reduce operations on the specified axes, including: min, max, sum, mean, product

API Reference please refer to:
:ref:`cn_api_fluid_layers_reduce_min`
:ref:`cn_api_fluid_layers_reduce_max`
:ref:`cn_api_fluid_layers_reduce_sum`
:ref:`cn_api_fluid_layers_reduce_mean`
:ref:`cn_api_fluid_layers_reduce_prod`


Binary operation
==================

elementwise_add
------------------

Add two :code:`Tensor` by element, the corresponding math operator is :code:`+`

API Reference See :ref:`cn_api_fluid_layers_elementwise_add`

elementwise_sub
------------------

Sub two :code:`Tensor` by element, the corresponding math operator is :code:`-`

API Reference See :ref:`cn_api_fluid_layers_elementwise_sub`

elementwise_mul
------------------

Multiply two :code:`Tensor` by element, the corresponding math operator is :code:`*`

API Reference See :ref:`cn_api_fluid_layers_elementwise_mul`

elementwise_div
------------------

Divide two :code:`Tensor` by element, the corresponding math operator is :code:`/` or :code:`//`

API Reference See :ref:`cn_api_fluid_layers_elementwise_div`


elementwise_pow
------------------

Do power operations on two :code:`Tensor` by element, the corresponding math operator is :code:`**`

API Reference please refer to :ref:`cn_api_fluid_layers_elementwise_pow`

equal
------------------

Judge whether the two :code:`Tensor` elements are equal, the corresponding math operator is :code:`==`

API Reference See :ref:`cn_api_fluid_layers_equal`


less_than
------------------

Judge whether the two :code:`Tensor` elements satisfy the less than relationship, the corresponding math operator is :code:`<`

API Reference See :ref:`cn_api_fluid_layers_less_than`



sum
------------------

Add two elements :code:`Tensor` by element.

API Reference please refer to :ref:`cn_api_fluid_layers_sum`

elementwise_min
------------------

Do :code:`min(x, y)` operations on two :code:`Tensor` by element .

API Reference please refer to :ref:`cn_api_fluid_layers_elementwise_min`

elementwise_max
------------------

Do :code:`max(x, y)` operations on two :code:`Tensor` by element .

API Reference See :ref:`cn_api_fluid_layers_elementwise_max`

matmul
------------------

Do matrix multiplication operations on two :code:`Tensor`.

API Reference please refer to :ref:`cn_api_fluid_layers_matmul`
