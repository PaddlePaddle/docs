隐式数据类型提升介绍
================

本篇文章主要为你介绍飞桨的隐式类型提升机制，帮助你更好的使用飞桨。

一、背景
--------

类型提升是当两个不同数据类型进行运算(+/-/*//)时，自动计算结果应得的数据类型，这个操作在不同类型数据之间进行数据计算时，是必不可少的一个功能。

类型提升计算时，根据调用分别分为运算符重载、二元 API：

-  运算符重载方法指通过运算符直接进行计算，如 a+b, a-b 等调用方法；

-  二元 API 则是通过 API 的形式对函数进行调用，如 paddle.add(a,b)等。

隐式数据类型提升即为在不需要用户进行额外操作的情况下，自动的帮用户进行类型提升的行为。

二、飞桨的类型提升规则
------------------------------

飞桨的类型提升的支持范围和规则在 Tensor 之间，Tensor 和 Scalar 之间会有所不同，以下将分别展开介绍。

1、 Tensor 之间的隐式类型提升规则介绍

-  Tensor 之间的自动类型提升仅支持浮点型之间的计算，以及复数和实数之间的计算，计算原则为返回两个 Tensor 中更大的数据类型，详情见下表：

+/-/* | bool | u8 | i8 | i16 | i32 | i64 | bf16 | f16 | f32 | f64 | c64 | c128 |
:-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: |
bool | - | - | - | - | - | - | - | - | - | - | c64 | c128 |
u8 | - | - | - | - | - | - | - | - | - | - | c64 | c128 |
i8 | - | - | - | - | - | - | - | - | - | - | c64 | c128 |
i16 | - | - | - | - | - | - | - | - | - | - | c64 | c128 |
i32 | - | - | - | - | - | - | - | - | - | - | c64 | c128 |
i64 | - | - | - | - | - | - | - | - | - | - | c64 | c128 |
bf16 | - | - | - | - | - | - | bf16 | f32 | f32 | f64 | c64 | c128 |
f16 | - | - | - | - | - | - | f32 | f16 | f32 | f64 | c64 | c128 |
f32 | - | - | - | - | - | - | f32 | f32 | f32 | f64 | c64 | c128 |
f64 | - | - | - | - | - | - | f64 | f64 | f64 | f64 | c128 | c128 |
c64 | c64 | c64 | c64 | c64 | c64 | c64 | c64 | c64 | c64 | c128 | c64 | c128 |
c128 | c128 | c128 | c128 | c128 | c128 | c128 | c128 | c128 | c128 | c128 | c128 | c128 |

-  以 paddle.add(a, b) 为例，上表中行为 a，列为 b。
-  对于逻辑性 API 来说，支持范围同上表，但结果均返回 bool 值。

2、 Tensor 与 Scalar 之间的隐式类型提升规则介绍

-  Tensor 与 Scalar 之间支持全类型自动类型提升，计算原则为在 Scalar 大类型一致（同为整型、浮点型等）时，向 Tensor 类型靠拢，否则与 Tensor 之间的计算结果一致。

-  其中视 int -> int64，float -> float32, bool -> bool, complex -> complex64。具体规则如下：

+/-/* | bool | int | float | complex |
:-----: | :-----: | :-----: | :-----: | :-----: |
bool | bool | i64 | f32 | c64 |
u8 | u8 | u8 | f32 | c64 |
i8 | i8 | i8 | f32 | c64 |
i16 | i16 | i16 | f32 | c64 |
i32 | i32 | i32 | f32 | c64 |
i64 | i64 | i64 | f32 | c64 |
bf16 | bf16 | bf16 | bf16 | c64 |
f16 | f16 | f16 | f16 | c64 |
f32 | f32 | f32 | f32 | c64 |
f64 | f64 | f64 | f64 | c128 |
c64 | c64 | c64 | c64 | c64 |
c128 | c128 | c128 | c128 | c128 |

-  对于逻辑性 API 来说，支持范围同上表，但结果均返回 bool 值。
-  对于特殊 API divide 来说，其不会返回比 float 更小的类型。如 int32 / Scalar 返回 float32 。

三、飞桨的隐式类型提升使用方法说明
------------------------------

.. code:: ipython3

    #加载飞桨和相关类库
    import paddle
    a = paddle.ones([3,3], dtype = 'float16')
    b = paddle.ones([3,3], dtype = 'float32')
    c = a + b # 此时将自动进行类型提升，将 a 的数据类型 cast 为 float32，不需要用户进行额外操作
    print (c.dtype)


.. parsed-literal::

    <VarType.FP32: 5>


.. code:: ipython3

    #加载飞桨和相关类库
    import paddle
    a = paddle.ones([3,3], dtype = 'int64')
    b = paddle.ones([3,3], dtype = 'float32')
    c = a + b # 此时由于不再支持 int 和 float 类型之间的自动类型提升，会报 Type Error

四、飞桨隐式类型提升的适用范围
------------------------------

截止到 Paddle 2.6 版本，具体支持的二元 API 及规则如下：

序号 | API | Tensor 之间的计算 | Tensor 和 Scalar 之间的计算 |
:-----: | :-----: | :-----: | :-----: |
1 | add | 通用规则 | 通用规则 |
2 | subtract | 通用规则 | 通用规则 |
3 | multiply | 通用规则 | 通用规则 |
4 | divide | 通用规则 | int to float |
5 | floor_divide | 通用规则 | 通用规则 |
6 | pow | 通用规则 | 通用规则 |
7 | equal | all bool | all bool |
8 | not_equal | all bool | all bool |
9 | less_than | all bool | all bool |
10 | less_equal | all bool | all bool |
11 | greater_than | all bool | all bool |
12 | greater_equal | all bool | all bool |
13 | logical_and | all bool | all bool |
14 | logical_or | all bool | all bool |
15 | logical_xor | all bool | all bool |
16 | bitwise_and | - | 通用规则 |
17 | bitwise_or | - | 通用规则 |
18 | bitwise_xor | - | 通用规则 |
19 | where | 通用规则 | 通用规则 |
20 | fmax | 通用规则 | - |
21 | fmin | 通用规则 | - |
22 | logaddexp | 通用规则 | - |
23 | maximum | 通用规则 | - |
24 | minimum | 通用规则 | - |
25 | remainder(mod) | 通用规则 | 通用规则 |
26 | huber_loss | 通用规则 | - |
27 | nextafter | 通用规则 | - |
28 | atan2 | 通用规则 | - |
29 | poisson_nll_loss | 通用规则 | - |
30 | l1_loss | 通用规则 | - |
31 | huber_loss | 通用规则 | - |
32 | mse_loss | 通用规则 | - |

五、总结
------------------------

本文章主要介绍了如何使用飞桨的隐式类型提升机制，以及如何使用飞桨的隐式类型提升。
