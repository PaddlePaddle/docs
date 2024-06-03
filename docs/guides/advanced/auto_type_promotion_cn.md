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

 ```python

    import paddle

    a = paddle.rand([3,3], dtype = 'float16')
    b = paddle.rand([3,3], dtype = 'float32')
    print (a + b) # 当 a 和 b 均为 Tensor 时，视为 Tensor 之间的计算

    a = paddle.rand([3,3], dtype = 'float16')
    b = 1.0
    print (a + b) # 当 a 和 b 其中任意一个为 Scalar 时，视为 Tensor 和 Scalar 之间的计算

 ```

1、 Tensor 之间的隐式类型提升规则介绍

-  由于模型训练中通常不会出现浮点型以外的不同类型之间的计算，为了更方便用户能够快速排查由于类型导致的问题，Tensor 之间的自动类型提升将仅支持浮点型之间的计算，以及复数和实数之间的计算，计算原则为返回两个 Tensor 中更大的数据类型，详情见下表：

+/-/* | bf16 | f16 | f32 | f64 | bool | u8 | i8 | i16 | i32 | i64 | c64 | c128 |
:-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: |
bf16 | bf16 | f32 | f32 | f64 | - | - | - | - | - | - | c64 | c128 |
f16 | f32 | f16 | f32 | f64 | - | - | - | - | - | - | c64 | c128 |
f32 | f32 | f32 | f32 | f64 | - | - | - | - | - | - | c64 | c128 |
f64 | f64 | f64 | f64 | f64 | - | - | - | - | - | - | c128 | c128 |
bool | - | - | - | - | - | - | - | - | - | - | c64 | c128 |
u8 | - | - | - | - | - | - | - | - | - | - | c64 | c128 |
i8 | - | - | - | - | - | - | - | - | - | - | c64 | c128 |
i16 | - | - | - | - | - | - | - | - | - | - | c64 | c128 |
i32 | - | - | - | - | - | - | - | - | - | - | c64 | c128 |
i64 | - | - | - | - | - | - | - | - | - | - | c64 | c128 |
c64 | c64 | c64 | c64 | c64 | c64 | c64 | c64 | c64 | c64 | c128 | c64 | c128 |
c128 | c128 | c128 | c128 | c128 | c128 | c128 | c128 | c128 | c128 | c128 | c128 | c128 |


-  以 paddle.add(a, b) 为例，上表中行为 a，列为 b。

 ```python

    import paddle

    # 浮点数间的计算
    a = paddle.rand([3,3], dtype = 'float16')
    b = paddle.rand([3,3], dtype = 'float32')
    c = a + b # 此时将自动进行类型提升，将 a 的数据类型 cast 为 float32，不需要用户进行额外操作
    print (c.dtype) # 输出类型为 float32

    a = paddle.rand([3,3], dtype = 'bfloat16')
    b = paddle.rand([3,3], dtype = 'float64')
    c = a + b # 此时将自动进行类型提升，将 a 的数据类型 cast 为 float64，不需要用户进行额外操作
    print (c.dtype) # 输出类型为 float64

    # 复数与实数之间的计算
    a = paddle.ones([3,3], dtype = 'complex64')
    b = paddle.rand([3,3], dtype = 'float64')
    c = a + b # 此时将自动进行类型提升，将 a 和 b 的数据类型 cast 为 complex128，不需要用户进行额外操作
    print (c.dtype) # 输出类型为 complex128

    # 复数之间的计算
    a = paddle.ones([3,3], dtype = 'complex128')
    b = paddle.ones([3,3], dtype = 'complex64')
    c = a + b # 此时将自动进行类型提升，将 b 的数据类型 cast 为 complex128，不需要用户进行额外操作
    print (c.dtype) # 输出类型为 complex128

 ```

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


 ```python

    import paddle

    # 当 Scalar 在大类型上与 Tensor 类型一致时，结果返回 Tensor 的类型
    a = paddle.rand([3,3], dtype = 'float16')
    b = 1.0
    c = a + b # a 与 b 大类型一致，都为 float 类型，因此将 b 自动 cast 为 float16，不需要用户进行额外操作
    print (c.dtype) # 输出类型为 float16

    # 当 Scalar 在大类型上与 Tensor 类型不一致时，遵循 Tensor 之间的计算规则
    a = 1.0
    b = paddle.ones([3,3], dtype = 'int64')
    c = a + b # a 与 b 大类型不一致，将 b 自动 cast 为 float32，不需要用户进行额外操作
    print (c.dtype) # 输出类型为 float32

 ```

三、飞桨的隐式类型提升使用方法说明
------------------------------

1、对于支持隐式类型提升的情况

 ```python

    import paddle
    a = paddle.rand([3,3], dtype = 'float16')
    b = paddle.rand([3,3], dtype = 'float32')
    c = a + b # 此时将自动进行类型提升，将 a 的数据类型 cast 为 float32，不需要用户进行额外操作
    print (c.dtype) # float32

    # 符合交换律
    d = b + a
    print (d.dtype) # float32
    print (paddle.allclose(c, d)) # Tensor(shape=[], dtype=bool, place=Place(gpu:0), stop_gradient=True, True)

    # 与二元 API 计算结果一致
    e = paddle.add(a, b)
    print (e.dtype) # float32
    print (paddle.allclose(c, e)) # Tensor(shape=[], dtype=bool, place=Place(gpu:0), stop_gradient=True, True)

    # 与静态图计算结果一致
    paddle.enable_static()
    exe = paddle.static.Executor()
    train_program = paddle.static.Program()
    startup_program = paddle.static.Program()
    with paddle.static.program_guard(train_program, startup_program):
        a = paddle.rand([3,3], dtype = 'float16')
        b = paddle.rand([3,3], dtype = 'float32')
        f = paddle.add(a, b)
        res = exe.run(train_program, fetch_list=[f])
    print (res[0].dtype) # float32
    paddle.disable_static()
    print (paddle.allclose(c, paddle.to_tensor(res[0]))) # Tensor(shape=[], dtype=bool, place=Place(gpu:0), stop_gradient=True, True)

 ```

2、对于不支持隐式类型提升的情况

 ```python

    import paddle
    a = paddle.ones([3,3], dtype = 'int64')
    b = paddle.rand([3,3], dtype = 'float32')
    c = a + b # 此时由于不再支持 int 和 float 类型之间的自动类型提升，会报 Type Error

    # 对于不支持自动类型提升的情况，建议用户进行手动进行类型提升
    # 方法一：使用 astype
    a = a.astype("float32")
    a = a.astype(b.dtype)

    # 方法二：使用 cast
    a = paddle.cast(a, "float32")
    a = paddle.cast(a, b.dtype)

 ```

四、飞桨隐式类型提升的适用范围
------------------------------

截止到 Paddle 2.6 版本，具体支持的二元 API 及规则如下：

序号 | API | Tensor 之间的计算 | Tensor 和 Scalar 之间的计算 |
:-----: | :-----: | :-----: | :-----: |
1 | add | 通用规则 | 通用规则 |
2 | subtract | 通用规则 | 通用规则 |
3 | multiply | 通用规则 | 通用规则 |
4 | divide | 通用规则 | 除法规则 |
5 | floor_divide | 通用规则 | 通用规则 |
6 | pow | 通用规则 | 通用规则 |
7 | equal | 逻辑规则 | 逻辑规则 |
8 | not_equal | 逻辑规则 | 逻辑规则 |
9 | less_than | 逻辑规则 | 逻辑规则 |
10 | less_equal | 逻辑规则 | 逻辑规则 |
11 | greater_than | 逻辑规则 | 逻辑规则 |
12 | greater_equal | 逻辑规则 | 逻辑规则 |
13 | logical_and | 逻辑规则 | 逻辑规则 |
14 | logical_or | 逻辑规则 | 逻辑规则 |
15 | logical_xor | 逻辑规则 | 逻辑规则 |
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

上表中存在两种特殊规则：

-  特殊规则-除法规则： 对于特殊 API divide 来说，其不会返回比 float 更小的类型。如 int32 / Scalar 返回 float32 。

 ```python

    import paddle
    a = paddle.ones([3,3], dtype = 'int32')
    b = 1
    c = a / b
    print (c.dtype) # float32

 ```

-  特殊规则-逻辑规则： 对于逻辑性 API 来说，由于复数类型无法直接进行逻辑运算，因此复数相关计算不在类型提升的支持范围内，在支持范围内其结果均返回 bool 值。

 ```python

    import paddle
    a = paddle.rand([3,3], dtype = 'float32')
    b = paddle.rand([3,3], dtype = 'float16')
    c = a == b
    print (c.dtype) # bool

 ```

五、总结
------------------------

Paddle 在支持隐式类型提升的基础上保证了计算符合交换律，运算符重载与二元 API 结果统一，动态图与静态图结果统一。本篇文章中明确了隐式类型提升的规则和支持范围，在介绍使用方法的同时根据当前版本总结了支持类型提升的二元 API 种类，希望能够借此增加用户在使用 Paddle 时的便利性。
