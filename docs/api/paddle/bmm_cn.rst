.. _cn_api_paddle_bmm:

bmm
-------------------------------

.. py:function:: paddle.bmm(x, y, out_dtype=None, name=None, *, out=None)




对输入 x 及输入 y 进行矩阵相乘。

两个输入的维度必须等于 3，并且矩阵 x 和矩阵 y 的第一维必须相等。同时矩阵 x 的第三维必须等于矩阵 y 的第二维。

例如：若 x 和 y 分别为 (b, m, k) 和 (b, k, n) 的矩阵，则函数的输出为一个 (b, m, n) 的矩阵。

参数
:::::::::

    - **x** (Tensor) - 第一个输入 Tensor。别名 ``input``。
    - **y** (Tensor) - 第二个输入 Tensor。别名 ``mat2``。
    - **out_dtype** (paddle.dtype|None，可选) - 输出的数据类型。目前仅支持在动态图中将 CUDA 上数据类型为 float16 或 bfloat16 的输入转换为 ``paddle.float32`` 输出，两个输入 Tensor 的数据类型必须相同。为保持向后兼容性，当恰好传入三个位置参数且第三个参数为字符串时，该参数会被视为 ``name``。如需以位置参数传入 ``out_dtype``，请使用 ``paddle.float32`` 等数据类型对象；若使用字符串数据类型，请通过 ``out_dtype="float32"`` 传入。当第 4 个位置参数提供 ``name`` 时，字符串数据类型也可作为第 3 个位置参数传入，例如 ``paddle.bmm(x, y, "float32", "my_name")``。默认值为 None。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

关键字参数
:::::::::

    - **out** (Tensor，可选) - 输出 Tensor，若不为 ``None``，计算结果将保存在该 Tensor 中，默认值为 ``None``。

返回
:::::::::
Tensor，批量矩阵相乘后的结果。未指定 ``out_dtype`` 时，输出的数据类型与输入相同。

代码示例
:::::::::

COPY-FROM: paddle.bmm
