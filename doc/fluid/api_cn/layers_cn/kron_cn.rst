.. _cn_api_paddle_tensor_kron:

kron
-------------------------------

.. py:function:: paddle.tensor.kron(x, y, out=None, name=None)


Kronecker Product 算子。

该 OP 计算两个张量的克罗内克积，结果是一个合成的张量，由第二个张量经过第一个张量中的元素缩放
后的组块构成。


这个 OP 预设两个张量 $X$ 和 $Y$ 的秩 (rank) 相同，如有必要，将会在秩较小的张量的形状前面补
上 1。令 $X$ 的形状是 [$r_0$, $r_1$, ..., $r_N$]，$Y$ 的形状是 
[$s_0$, $s_1$, ..., $s_N$]，那么输出张量的形状是 
[$r_{0}s_{0}$, $r_{1}s_{1}$, ..., $r_{N}s_{N}$]. 其中的元素是 $X$ 和 $Y$ 中的元素
的乘积。

公式为

.. math::

          output[k_{0}, k_{1}, ..., k_{N}] = X[i_{0}, i_{1}, ..., i_{N}] *
          Y[j_{0}, j_{1}, ..., j_{N}]


其中

.. math::

          k_{t} = i_{t} * s_{t} + j_{t}, t = 0, 1, ..., N


参数:
  - **x** (Variable) – Kron OP 的第一个输入。多维 Tensor，数据类型为 float16, float32, float64, int32 或 int64。
  - **y** (Variable) – Kron OP 的第二个输入。多维 Tensor，数据类型为 float16, float32, float64, int32 或 int64，与 x 相同。
  - **out**  (Variable， 可选) -  指定算子输出结果的 Tensor，可以是程序中已经创建的任何 Variable。默认值为 None，此时将创建新的 Variable 来保存输出结果。
  - **name** (str，可选) – 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` ，默认值为 None。

返回：
  - Kron OP 的输出。多维 Tensor，数据类型为 float16, float32, float64, int32 或 int64，与 x 一致。

返回类型: Variable 


**代码示例**

..  code-block:: python

  import paddle
  from paddle import fluid
  import paddle.fluid.dygraph as dg
  import numpy as np

  a = np.arange(1, 5).reshape(2, 2).astype(np.float32)
  b = np.arange(1, 10).reshape(3, 3).astype(np.float32)

  place = fluid.CPUPlace()
  with dg.guard(place):
      a_var = dg.to_variable(a)
      b_var = dg.to_variable(b)
      c_var = paddle.kron(a_var, b_var)
      c_np = c_var.numpy()
  print(c_np)

  #[[ 1.  2.  3.  2.  4.  6.]
  # [ 4.  5.  6.  8. 10. 12.]
  # [ 7.  8.  9. 14. 16. 18.]
  # [ 3.  6.  9.  4.  8. 12.]
  # [12. 15. 18. 16. 20. 24.]
  # [21. 24. 27. 28. 32. 36.]]
