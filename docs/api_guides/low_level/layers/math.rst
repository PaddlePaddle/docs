..  _api_guide_math:


数学操作
#########

Paddle 提供了丰富的数学操作，以下列出的数学操作都是对目标张量进行逐元素的操作。其中，如果二元操作的两个输入有不同形状，会先进行 :code:`broadcast`. 部分数学操作还支持数学操作符，比如： :code:`+`,  :code:`-`, :code:`*`, :code:`/` 等。数学操作符不仅支持张量，还支持标量。


一元操作
==================

exp
------------------

对输入 :code:`Tensor` 逐元素做 :code:`exp` 操作。

API Reference 请参考 :ref:`cn_api_fluid_layers_exp`

tanh
------------------

对输入 :code:`Tensor` 逐元素取正切。

API Reference 请参考 :ref:`cn_api_fluid_layers_tanh`

sqrt
------------------

对输入 :code:`Tensor` 逐元素取平方根。

API Reference 请参考 :ref:`cn_api_fluid_layers_sqrt`

abs
------------------

对输入 :code:`Tensor` 逐元素取绝对值。

API Reference 请参考 :ref:`cn_api_fluid_layers_abs`

ceil
------------------

对输入 :code:`Tensor` 逐元素向上取整。

API Reference 请参考 :ref:`cn_api_fluid_layers_ceil`

floor
------------------

对输入 :code:`Tensor` 逐元素向下取整。

API Reference 请参考 :ref:`cn_api_fluid_layers_floor`

sin
------------------

对输入 :code:`Tensor` 逐元素取正弦。

API Reference 请参考 :ref:`cn_api_fluid_layers_sin`

cos
------------------

对输入 :code:`Tensor` 逐元素取余弦。

API Reference 请参考 :ref:`cn_api_fluid_layers_cos`

cosh
------------------

对输入 :code:`Tensor` 逐元素取双曲余弦。

API Reference 请参考 :ref:`cn_api_fluid_layers_cosh`

round
------------------

对输入 :code:`Tensor` 逐元素四舍五入取整。

API Reference 请参考 :ref:`cn_api_fluid_layers_round`

square
------------------

对输入 :code:`Tensor` 逐元素取平方。

API Reference 请参考 :ref:`cn_api_fluid_layers_square`

reciprocal
------------------

对输入 :code:`Tensor` 逐元素取倒数。

API Reference 请参考 :ref:`cn_api_fluid_layers_reciprocal`


reduce
------------------

对输入 :code:`Tensor` 在指定的若干轴上做 reduce 操作，包括：min, max, sum, mean, product

API Reference 请参考:
:ref:`cn_api_fluid_layers_reduce_min`
:ref:`cn_api_fluid_layers_reduce_max`
:ref:`cn_api_fluid_layers_reduce_sum`
:ref:`cn_api_fluid_layers_reduce_mean`
:ref:`cn_api_fluid_layers_reduce_prod`


二元操作
==================

elementwise_add
------------------

对两个 :code:`Tensor` 逐元素相加，对应的数学操作符为 :code:`+`

API Reference 请参考 :ref:`cn_api_fluid_layers_elementwise_add`

elementwise_sub
------------------

对两个 :code:`Tensor` 逐元素相减，对应数学操作符 :code:`-`

API Reference 请参考 :ref:`cn_api_fluid_layers_elementwise_sub`

elementwise_mul
------------------

对两个 :code:`Tensor` 逐元素相乘， 对应数学操作符 :code:`*`

API Reference 请参考 :ref:`cn_api_fluid_layers_elementwise_mul`

elementwise_div
------------------

对两个 :code:`Tensor` 逐元素相除， 对应数学操作符 :code:`/` 或 :code:`//`

API Reference 请参考 :ref:`cn_api_fluid_layers_elementwise_div`


elementwise_pow
------------------

对两个 :code:`Tensor` 逐元素做次幂操作， 对应数学操作符 :code:`**`

API Reference 请参考 :ref:`cn_api_fluid_layers_elementwise_pow`

equal
------------------

对两个 :code:`Tensor` 逐元素判断是否相等， 对应数学操作符 :code:`==`

API Reference 请参考 :ref:`cn_api_fluid_layers_equal`


less_than
------------------

对两个 :code:`Tensor` 逐元素判断是否满足小于关系， 对应数学操作符 :code:`<`

API Reference 请参考 :ref:`cn_api_fluid_layers_less_than`



sum
------------------

对两个 :code:`Tensor` 逐元素相加。

API Reference 请参考 :ref:`cn_api_fluid_layers_sum`

elementwise_min
------------------

对两个 :code:`Tensor` 逐元素进行 :code:`min(x, y)` 操作。

API Reference 请参考 :ref:`cn_api_fluid_layers_elementwise_min`

elementwise_max
------------------

对两个 :code:`Tensor` 逐元素进行 :code:`max(x, y)` 操作。

API Reference 请参考 :ref:`cn_api_fluid_layers_elementwise_max`

matmul
------------------

对两个 :code:`Tensor` 进行矩阵乘操作。

API Reference 请参考 :ref:`cn_api_fluid_layers_matmul`
