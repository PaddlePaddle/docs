.. _cn_api_tensor_random_randint_like:

randint_like
-------------------------------

.. py:function:: paddle.randint_like(x, low=0, high=None, dtype=None, name=None)

该OP返回服从均匀分布的、范围在[``low``, ``high``)的随机Tensor，输出的形状与x的形状一致，当数据类型 ``dtype`` 为None时（默认），输出的数据类型与x的数据类型一致，当数据类型 ``dtype`` 不为None时，将输出用户指定的数据类型。当 ``high`` 为None时（默认），均匀采样的区间为[0, ``low``)。

参数
::::::::::
    - **x** (Tensor) – 输入的多维Tensor，数据类型可以是bool，int32，int64，float16，float32，float64。输出Tensor的形状和 ``x`` 相同。如果 ``dtype`` 为None，则输出Tensor的数据类型与 ``x`` 相同。
    - **low** (int) - 要生成的随机值范围的下限，``low`` 包含在范围中。当 ``high`` 为None时，均匀采样的区间为[0, ``low``)。默认值为0。
    - **high** (int, 可选) - 要生成的随机值范围的上限，``high`` 不包含在范围中。默认值为None，此时范围是[0, ``low``)。
    - **dtype** (str|np.dtype, 可选) - 输出Tensor的数据类型，支持bool，int32，int64，float16，float32，float64。当该参数值为None时， 输出Tensor的数据类型与输入Tensor的数据类型一致。默认值为None。
    - **name** (str, 可选) - 输出的名字。一般无需设置，默认值为None。该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` 。

返回
::::::::::
    Tensor：从区间[``low``，``high``)内均匀分布采样的随机Tensor，形状为 ``x.shape``，数据类型为 ``dtype``。

代码示例
:::::::::::

.. code-block:: python

    import paddle

    # example 1:
    # dtype is None and the dtype of x is float16
    x = paddle.zeros((1,2)).astype("float16")
    out1 = paddle.randint_like(x, low=-5, high=5)
    print(out1)
    print(out1.dtype)
    # [[0, -3]]  # random
    # paddle.float16

    # example 2:
    # dtype is None and the dtype of x is float32
    x = paddle.zeros((1,2)).astype("float32")
    out2 = paddle.randint_like(x, low=-5, high=5)
    print(out2)
    print(out2.dtype)
    # [[0, -3]]  # random
    # paddle.float32

    # example 3:
    # dtype is None and the dtype of x is float64
    x = paddle.zeros((1,2)).astype("float64")
    out3 = paddle.randint_like(x, low=-5, high=5)
    print(out3)
    print(out3.dtype)
    # [[0, -3]]  # random
    # paddle.float64

    # example 4:
    # dtype is None and the dtype of x is int32
    x = paddle.zeros((1,2)).astype("int32")
    out4 = paddle.randint_like(x, low=-5, high=5)
    print(out4)
    print(out4.dtype)
    # [[0, -3]]  # random
    # paddle.int32

    # example 5:
    # dtype is None and the dtype of x is int64
    x = paddle.zeros((1,2)).astype("int64")
    out5 = paddle.randint_like(x, low=-5, high=5)
    print(out5)
    print(out5.dtype)
    # [[0, -3]]  # random
    # paddle.int64

    # example 6:
    # dtype is float64 and the dtype of x is float32
    x = paddle.zeros((1,2)).astype("float32")
    out6 = paddle.randint_like(x, low=-5, high=5, dtype="float64")
    print(out6)
    print(out6.dtype)
    # [[0, -1]]  # random
    # paddle.float64

    # example 7:
    # dtype is bool and the dtype of x is float32
    x = paddle.zeros((1,2)).astype("float32")
    out7 = paddle.randint_like(x, low=-5, high=5, dtype="bool")
    print(out7)
    print(out7.dtype)
    # [[0, -1]]  # random
    # paddle.bool

    # example 8:
    # dtype is int32 and the dtype of x is float32
    x = paddle.zeros((1,2)).astype("float32")
    out8 = paddle.randint_like(x, low=-5, high=5, dtype="int32")
    print(out8)
    print(out8.dtype)
    # [[0, -1]]  # random
    # paddle.int32

    # example 9:
    # dtype is int64 and the dtype of x is float32
    x = paddle.zeros((1,2)).astype("float32")
    out9 = paddle.randint_like(x, low=-5, high=5, dtype="int64")
    print(out9)
    print(out9.dtype)
    # [[0, -1]]  # random
    # paddle.int64

    # example 10:
    # dtype is int64 and the dtype of x is bool
    x = paddle.zeros((1,2)).astype("bool")
    out10 = paddle.randint_like(x, low=-5, high=5, dtype="int64")
    print(out10)
    print(out10.dtype)
    # [[0, -1]]  # random
    # paddle.int64