.. _cn_api_paddle_kron:

kron
-------------------------------

.. py:function:: paddle.kron(x, y, out=None, name=None)





Kronecker Product 算子。

计算两个 Tensor 的克罗内克积，结果是一个合成的 Tensor，由第二个 Tensor 经过第一个 Tensor 中的元素缩放后的组块构成。


预设两个 Tensor $X$ 和 $Y$ 的秩 (rank) 相同，如有必要，将会在秩较小的 Tensor 的形状前面补上 1。令 $X$ 的形状是 [$r_0$, $r_1$, ..., $r_N$]，$Y$ 的形状是
[$s_0$, $s_1$, ..., $s_N$]，那么输出 Tensor 的形状是 [$r_{0}s_{0}$, $r_{1}s_{1}$, ..., $r_{N}s_{N}$]，其中的元素是 $X$ 和 $Y$ 中的元素的乘积。

公式为

.. math::

          output[k_{0}, k_{1}, ..., k_{N}] = X[i_{0}, i_{1}, ..., i_{N}] *
          Y[j_{0}, j_{1}, ..., j_{N}]


其中

.. math::

          k_{t} = i_{t} * s_{t} + j_{t}, t = 0, 1, ..., N


参数
::::::::::::

  - **x** (Tensor) – Kron OP 的第一个输入。多维 Tensor，数据类型为 float16、float32、float64、int32 或 int64。
  - **y** (Tensor) – Kron OP 的第二个输入。多维 Tensor，数据类型为 float16、float32、float64、int32 或 int64，与 x 相同。
  - **out**  (Tensor，可选) -  指定算子输出结果的 Tensor，可以是程序中已经创建的任何 Tensor。默认值为 None，此时将创建新的 Tensor 来保存输出结果。
  - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::

多维 Tensor，数据类型为 float16、float32、float64、int32 或 int64，与 x 一致。



代码示例
::::::::::::

COPY-FROM: paddle.kron
