.. _cn_api_paddle_linalg_norm:

norm
-------------------------------

.. py:function:: paddle.linalg.norm(x, p=None, axis=None, keepdim=False, name=None):




将计算给定 Tensor 的矩阵范数（Frobenius 范数, Nuclear 范数或 p 范数）和向量范数（向量 1 范数、2 范数、或者通常的 p 范数）。

该函数计算的是向量范数还是矩阵范数，确定方法如下:
- 如果 axis 是 int 类型，计算向量范数
- 如果 axis 是二维数组，计算矩阵范数
- 如果 axis 为 None，x 会被压缩成一维向量然后计算向量范数


**Paddle 支持以下范数:**

+----------------+--------------------------------+--------------------------------+
|     porder     |        norm for matrices       |        norm for vectors        |
+================+================================+================================+
| None(default)  |         frobenius norm         |              2 范数              |
+----------------+--------------------------------+--------------------------------+
|       fro      |         frobenius norm         |             不支持              |
+----------------+--------------------------------+--------------------------------+
|       nuc      |          nuclear norm          |             不支持              |
+----------------+--------------------------------+--------------------------------+
|       inf      |     max(sum(abs(x), dim=1))    |            max(abs(x))         |
+----------------+--------------------------------+--------------------------------+
|      -inf      |     min(sum(abs(x), dim=1))    |            min(abs(x))         |
+----------------+--------------------------------+--------------------------------+
|       0        |            不支持               |            sum(x != 0)         |
+----------------+--------------------------------+--------------------------------+
|       1        |     max(sum(abs(x), dim=0))    |              同下               |
+----------------+--------------------------------+--------------------------------+
|      -1        |     min(sum(abs(x), dim=0))    |              同下               |
+----------------+--------------------------------+--------------------------------+
|       2        |     由 axis 组成矩阵的最大奇异值   |              同下              |
+----------------+--------------------------------+--------------------------------+
|      -2        |     由 axis 组成矩阵的最大奇异值   |              同下              |
+----------------+--------------------------------+--------------------------------+
|    其他 int 或  |            不支持               | sum(abs(x)^{porder})^          |
|     float      |                                | {(1 / porder)}                 |
+----------------+--------------------------------+--------------------------------+


参数
:::::::::

    - **x** (Tensor) - 输入 Tensor。维度为多维，数据类型为 float32 或 float64。
    - **p** (int|float|string，可选) - 范数(ord)的种类。目前支持的值为 `fro`、`nuc`、`inf`、`-inf`、`0`、`1`、`2`，和任何实数 p 对应的 p 范数。默认值为 None。
    - **axis** (int|list|tuple，可选) - 使用范数计算的轴。如果 ``axis`` 为 None，则忽略 input 的维度，将其当做向量来计算。如果 ``axis`` 为 int 或者只有一个元素的 list|tuple，``norm`` API 会计算输入 Tensor 的向量范数。如果 axis 为包含两个元素的 list，API 会计算输入 Tensor 的矩阵范数。当 ``axis < 0`` 时，实际的计算维度为 rank(input) + axis。默认值为 `None` 。
    - **keepdim** (bool，可选) - 是否在输出的 Tensor 中保留和输入一样的维度，默认值为 False。当 :attr:`keepdim` 为 False 时，输出的 Tensor 会比输入 :attr:`input` 的维度少一些。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::

 Tensor，在指定 axis 上进行范数计算的结果，与输入 input 数据类型相同。

代码示例
:::::::::

COPY-FROM: paddle.linalg.norm
