.. _cn_api_tensor_linalg_dist:

dist
-------------------------------

.. py:function:: paddle.dist(x, y, p=2, name=None)

计算 `(x-y)` 的 p 范数（p-norm），需要注意这不是严格意义上的范数，仅作为距离的度量。输入 `x` 和 `y` 的形状（shape）必须是可广播的（broadcastable）。其含义如下，详情请参考 `Tensor 的广播机制 <../../guides/beginner/tensor_cn.html#id7>`_ ：

- 每个输入都至少有 1 维
- 对两个输入的维度从后向前匹配，两个输入每一维的大小需要满足 3 个条件中的任意一个：相等、其中一个为 1 或者其中一个不存在。

定义 `z = x - y` ，`x` 和 `y` 的形状是可广播的，那么 `z` 的形状可按照下列步骤得到：

(1) 如果 `x` 和 `y` 的维数不同，先对维数较少的这个输入的维度往前补 1。

例如，`x` 的形状为[8, 1, 6, 1]，`y` 的形状为[7, 1, 5]，对 `y` 的维度补 1，

x (4-D Tensor):  8 x 1 x 6 x 1

y (4-D Tensor):  1 x 7 x 1 x 5


(2) 确定输出 `z` 每一维度的大小：从两个输入的维度中选取最大值。

z (4-D Tensor):  8 x 7 x 6 x 5

若两个输入的维数相同，则输出的大小可直接用步骤 2 确定。以下是 `p` 取不同值时，范数的计算公式：

当 `p = 0`，定义 $0^0 = 0$，则 z 的零范数是 `z` 中非零元素的个数。

.. math::
    ||z||_{0}=\lim_{p \rightarrow 0}\sum_{i=1}^{m}|z_i|^{p}

当 `p = inf` ，`z` 的无穷范数是 `z` 所有元素中的绝对值最大值。

.. math::
    ||z||_\infty=\max_i |z_i|

当 `p = -inf` ，`z` 的负无穷范数是 `z` 所有元素中的绝对值最小值。

.. math::
    ||z||_{-\infty}=\min_i |z_i|

其他情况下，`z` 的 `p` 范数使用以下公式计算：

.. math::
    ||z||_{p}=(\sum_{i=1}^{m}|z_i|^p)^{\frac{1}{p}}

参数
::::::::::::

  - **x** (Tensor) - 1-D 到 6-D Tensor，数据类型为 float16, float32 或 float64。
  - **y** (Tensor) - 1-D 到 6-D Tensor，数据类型为 float16, float32 或 float64。
  - **p** (float，可选) - 用于设置需要计算的范数，数据类型为 float32 或 float64。默认值为 2。
  - **name** (str，可选) - 具体用法请参见  :ref:`api_guide_Name` ，一般无需设置，默认值为 None。



返回
::::::::::::
Tensor，`(x-y)` 的 `p` 范数。

代码示例
::::::::::::

COPY-FROM: paddle.dist
