Introduction to Data Type Promotion
================

This article introduces PaddlePaddle's data type promotion mechanism, aiding better usage of PaddlePaddle.

Background
--------

Type promotion automatically determines the resulting data type when performing operations (+/-/*//) on different data types. This essential function facilitates data calculations between varied types.

Type promotion calculations are categorized based on the call method:

-  Operator Overloading: Uses operators directly for computations, such as a + b or a - b.

-  Binary API: Uses API functions for computations, such as paddle.add(a, b).

Data type promotion automatically handles type promotion without user intervention.

Type Promotion Rules
------------------------------

The scope and rules of type promotion differ between Tensor-to-Tensor and Tensor-to-Scalar operations. The following will be introduced separately.

 ```python

    import paddle

    a = paddle.rand([3,3], dtype = 'float16')
    b = paddle.rand([3,3], dtype = 'float32')
    print (a + b) # when both a and b is tensor, treats as Tensor-to-Tensor

    a = paddle.rand([3,3], dtype = 'float16')
    b = 1.0
    print (a + b) # when either a or b is Scalar, treats as Tensor-to-Scalar

 ```

1. Type Promotion Rules in Tensor-to-Tensor

-  In model training, computations between different data types are usually limited to floating-point types. To help users quickly troubleshoot type-related issues, automatic type promotion between Tensors only supports floating-point types and calculations between complex and real numbers. The result type is the larger of the two input types. More Details show in this table:

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


-  Taking Paddle add (a, b) as an example, in the table above, line represent a and column represent b.

Sample Code:

 ```python

    import paddle

    # Calculation between floating points
    a = paddle.rand([3,3], dtype = 'float16')
    b = paddle.rand([3,3], dtype = 'float32')
    c = a + b # type promotion will automatically occur, casting 'a' to float32, and no additional user actions required
    print (c.dtype) # the dtype of 'c' is float32

    a = paddle.rand([3,3], dtype = 'bfloat16')
    b = paddle.rand([3,3], dtype = 'float64')
    c = a + b # type promotion will automatically occur, casting 'a' to float64, and no additional user actions required
    print (c.dtype) # the dtype of 'c' is float64

    # Calculation between complex and real number
    a = paddle.ones([3,3], dtype = 'complex64')
    b = paddle.rand([3,3], dtype = 'float64')
    c = a + b # type promotion will automatically occur, casting both 'a' and 'b' to complex128, and no additional user actions required
    print (c.dtype) # the dtype of 'c' is complex128

    # Calculation between complex numbers
    a = paddle.ones([3,3], dtype = 'complex128')
    b = paddle.ones([3,3], dtype = 'complex64')
    c = a + b # type promotion will automatically occur, casting 'b' to complex128, and no additional user actions required
    print (c.dtype) # the dtype of 'c' is complex128

 ```

2. Type Promotion Rules in Tensor-to-Scalar

-  Type promotion between Tensor and Scalar supports all types. The principle is to promote towards the Tensor's type when the Scalar's broad type (both are integers or both are floating-point, etc.) matches. Otherwise, the result follows the Tensor-to-Tensor rules.

-  The scalar operand has default dtype: int -> int64，float -> float32, bool -> bool, complex -> complex64. More Details show in this table:

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

Sample Code:

 ```python

    import paddle

    # Both Scalar and Tensor are floating-point, then return Tensor's type
    a = paddle.rand([3,3], dtype = 'float16')
    b = 1.0
    c = a + b # type promotion will automatically occur, casting 'b' to float16, and no additional user actions required
    print (c.dtype) # the dtype of 'c' is float16

    # Scalar and Tensor unmatch in broad type, the result follows the Tensor-to-Tensor rules
    a = 1.0
    b = paddle.ones([3,3], dtype = 'int64')
    c = a + b # type promotion will automatically occur, casting 'b' to float32, and no additional user actions required
    print (c.dtype) # the dtype of 'c' is float16

 ```

How to Use Type Promotion
------------------------------

1. For Supported Case

 ```python

    import paddle
    a = paddle.rand([3,3], dtype = 'float16')
    b = paddle.rand([3,3], dtype = 'float32')
    c = a + b # type promotion will automatically occur, casting 'a' to float32, and no additional user actions required
    print (c.dtype) # float32

    # Coincidence computative law
    d = b + a
    print (d.dtype) # float32
    print (paddle.allclose(c, d)) # Tensor(shape=[], dtype=bool, place=Place(gpu:0), stop_gradient=True, True)

    # Same with binary API
    e = paddle.add(a, b)
    print (e.dtype) # float32
    print (paddle.allclose(c, e)) # Tensor(shape=[], dtype=bool, place=Place(gpu:0), stop_gradient=True, True)

    # Same with static graph
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

2、For Unsupported Case

 ```python

    import paddle
    a = paddle.ones([3,3], dtype = 'int64')
    b = paddle.rand([3,3], dtype = 'float32')
    c = a + b # due to the unsupported of automatic type promotion between int and float, TypeError will be raised

    # For those unsupported cases, we suggest that users manually perform type promotion
    # method 1: use astype API
    a = a.astype("float32")
    a = a.astype(b.dtype)

    # method 2：use cast API
    a = paddle.cast(a, "float32")
    a = paddle.cast(a, b.dtype)

 ```

The Scope of Type Promotion
------------------------------

As of Paddle version 2.6, the supported binary APIs and their rules are as follows:

Number | API | Tensor-to-Tensor | Tensor-to-Scalar |
:-----: | :-----: | :-----: | :-----: |
1 | add | Common | Common |
2 | subtract | Common Rule | Common Rule |
3 | multiply | Common Rule | Common Rule |
4 | divide | Common Rule | Divide Rule |
5 | floor_divide | Common Rule | Common Rule |
6 | pow | Common Rule | Common Rule |
7 | equal | Logic Rule | Logic Rule |
8 | not_equal | Logic Rule | Logic Rule |
9 | less_than | Logic Rule | Logic Rule |
10 | less_equal | Logic Rule | Logic Rule |
11 | greater_than | Logic Rule | Logic Rule |
12 | greater_equal | Logic Rule | Logic Rule |
13 | logical_and | Logic Rule | Logic Rule |
14 | logical_or | Logic Rule | Logic Rule |
15 | logical_xor | Logic Rule | Logic Rule |
16 | bitwise_and | - | Common Rule |
17 | bitwise_or | - | Common Rule |
18 | bitwise_xor | - | Common Rule |
19 | where | Common Rule | Common Rule |
20 | fmax | Common Rule | - |
21 | fmin | Common Rule | - |
22 | logaddexp | Common Rule | - |
23 | maximum | Common Rule | - |
24 | minimum | Common Rule | - |
25 | remainder(mod) | Common Rule | Common Rule |
26 | huber_loss | Common Rule | - |
27 | nextafter | Common Rule | - |
28 | atan2 | Common Rule | - |
29 | poisson_nll_loss | Common Rule | - |
30 | l1_loss | Common Rule | - |
31 | huber_loss | Common Rule | - |
32 | mse_loss | Common Rule | - |

There are two specail rules in this table above:

-  Divide Rule: For divide API, it will not return dtype smaller than float. Such as int32 / Scalar returns float32.

 ```python

    import paddle
    a = paddle.ones([3,3], dtype = 'int32')
    b = 1
    c = a / b
    print (c.dtype) # float32

 ```

-  Logic Rule: For logical API, since complex types cannot be directly used for logical operations, calculations involving complex types are not within the scope of type promotion support. Within the supported scope, all results return bool type.

 ```python

    import paddle
    a = paddle.rand([3,3], dtype = 'float32')
    b = paddle.rand([3,3], dtype = 'float16')
    c = a == b
    print (c.dtype) # bool

 ```

Summary
------------------------

Paddle ensures that calculations comply with the commutative property while supporting data type promotion, with consistent results for operator overloading and binary APIs, as well as for dynamic and static graphs. This article clarifies the rules and scope of data type promotion, summarizes the types of binary APIs that support data type promotion in the current version, and aims to enhance user convenience when using PaddlePaddle by introducing usage methods.
