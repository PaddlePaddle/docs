..  _api_guide_math:


数学操作
#########

Paddle提供了丰富的数学操作，以下列出的数学操作都是对目标张量进行逐元素的操作。其中，如果二元操作的两个输入有不同形状，会先进行 :code:`broadcast`. 部分数学操作还支持数学操作符，比如： :code:`+`,  :code:`-`, :code:`*`, :code:`/` 等。数学操作符不仅支持张量，还支持标量。

elementwise_add
------------------

对两个 :code:`Tensor` 逐元素相加，对应的数学操作符为 :code:`+`

API Reference 请参考 api_fluid_math_elementwise_add_

elementwise_sub
------------------

对两个 :code:`Tensor` 逐元素相减，对应数学操作符 :code:`-`

API Reference 请参考 api_fluid_math_elementwise_sub_

elementwise_mul
------------------

对两个 :code:`Tensor` 逐元素相乘， 对应数学操作符 :code:`*`

API Reference 请参考 api_fluid_math_elementwise_mul_

elementwise_div
------------------

对两个 :code:`Tensor` 逐元素相除， 对应数学操作符 :code:`/` 或 :code:`//`

API Reference 请参考 api_fluid_math_elementwise_div_


elementwise_pow
------------------

对两个 :code:`Tensor` 逐元素做次幂操作， 对应数学操作符 :code:`**`

API Reference 请参考 api_fluid_math_elementwise_pow_

equal
------------------

对两个 :code:`Tensor` 逐元素判断是否相等， 对应数学操作符 :code:`==`

API Reference 请参考 api_fluid_math_equal_

not_equal
------------------

对两个 :code:`Tensor` 逐元素判断是否不等， 对应数学操作符 :code:`!=`

API Reference 请参考 api_fluid_math_elementwise_not_equal_

less_than
------------------

对两个 :code:`Tensor` 逐元素判断是否满足小于关系， 对应数学操作符 :code:`<`

API Reference 请参考 api_fluid_math_less_than_

less_equal
------------------

对两个 :code:`Tensor` 逐元素判断是否满足小于或等于关系， 对应数学操作符 :code:`<=`

API Reference 请参考 api_fluid_math_less_equal_

greater_than
------------------

对两个 :code:`Tensor` 逐元素判断是否满足大于关系， 对应数学操作符 :code:`>`

API Reference 请参考 api_fluid_math_greater_than_

greater_equal
------------------

对两个 :code:`Tensor` 逐元素判断是否满足大于或等于关系， 对应数学操作符 :code:`>=`

API Reference 请参考 api_fluid_math_greater_equal_

exp
------------------

对输入 :code:`Tensor` 逐元素做 :code:`exp` 操作。

API Reference 请参考 api_fluid_math_exp_

tanh
------------------

对输入 :code:`Tensor` 逐元素取正切。

API Reference 请参考 api_fluid_math_tanh_

sqrt
------------------

对输入 :code:`Tensor` 逐元素取平方根。

API Reference 请参考 api_fluid_math_sqrt_

abs
------------------

对输入 :code:`Tensor` 逐元素取绝对值。

API Reference 请参考 api_fluid_math_abs_

ceil
------------------

对输入 :code:`Tensor` 逐元素向上取整。

API Reference 请参考 api_fluid_math_ceil_

floor
------------------

对输入 :code:`Tensor` 逐元素向下取整。

API Reference 请参考 api_fluid_math_floor_

sin
------------------

对输入 :code:`Tensor` 逐元素取正玄。

API Reference 请参考 api_fluid_math_sin_

cos
------------------

对输入 :code:`Tensor` 逐元素取余玄。

API Reference 请参考 api_fluid_math_cos_

round
------------------

对输入 :code:`Tensor` 逐元素四舍五入取整。

API Reference 请参考 api_fluid_math_round_

square
------------------

对输入 :code:`Tensor` 逐元素取平方。

API Reference 请参考 api_fluid_math_square_

reciprocal
------------------

对输入 :code:`Tensor` 逐元素取倒数。

API Reference 请参考 api_fluid_math_reciprocal_

.. _api_fluid_math_elementwise_add: http://www.paddlepaddle.org/documentation/api/zh/1.0/layers.html#elementwise-add
.. _api_fluid_math_elementwise_sub: http://www.paddlepaddle.org/documentation/api/zh/1.0/layers.html#elementwise-sub
.. _api_fluid_math_elementwise_mul: http://www.paddlepaddle.org/documentation/api/zh/1.0/layers.html#elementwise-mul
.. _api_fluid_math_elementwise_div: http://www.paddlepaddle.org/documentation/api/zh/1.0/layers.html#elementwise-div
.. _api_fluid_math_elementwise_pow: http://www.paddlepaddle.org/documentation/api/zh/1.0/layers.html#elementwise-pow
.. _api_fluid_math_equal: http://www.paddlepaddle.org/documentation/api/zh/1.0/layers.html#equal
.. _api_fluid_math_not_equal: http://www.paddlepaddle.org/documentation/api/zh/1.0/layers.html#not-equal
.. _api_fluid_math_less_than: http://www.paddlepaddle.org/documentation/api/zh/1.0/layers.html#less-than
.. _api_fluid_math_less_equal: http://www.paddlepaddle.org/documentation/api/zh/1.0/layers.html#less-equal
.. _api_fluid_math_greater_than: http://www.paddlepaddle.org/documentation/api/zh/1.0/layers.html#greater-than
.. _api_fluid_math_greater_equal: http://www.paddlepaddle.org/documentation/api/zh/1.0/layers.html#greater-equal
.. _api_fluid_math_exp: http://www.paddlepaddle.org/documentation/api/zh/1.0/layers.html#exp
.. _api_fluid_math_tanh: http://www.paddlepaddle.org/documentation/api/zh/1.0/layers.html#tanh
.. _api_fluid_math_sqrt: http://www.paddlepaddle.org/documentation/api/zh/1.0/layers.html#sqrt
.. _api_fluid_math_abs: http://www.paddlepaddle.org/documentation/api/zh/1.0/layers.html#abs
.. _api_fluid_math_ceil: http://www.paddlepaddle.org/documentation/api/zh/1.0/layers.html#ceil
.. _api_fluid_math_floor: http://www.paddlepaddle.org/documentation/api/zh/1.0/layers.html#floor
.. _api_fluid_math_sin: http://www.paddlepaddle.org/documentation/api/zh/1.0/layers.html#sin
.. _api_fluid_math_cos: http://www.paddlepaddle.org/documentation/api/zh/1.0/layers.html#cos
.. _api_fluid_math_round: http://www.paddlepaddle.org/documentation/api/zh/1.0/layers.html#round
.. _api_fluid_math_square: http://www.paddlepaddle.org/documentation/api/zh/1.0/layers.html#square
.. _api_fluid_math_reciprocal: http://www.paddlepaddle.org/documentation/api/zh/1.0/layers.html#reciprocal
