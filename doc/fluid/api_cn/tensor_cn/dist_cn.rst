.. _cn_api_tensor_linalg_dist:

dist
-------------------------------

.. py:function:: paddle.tensor.linalg.dist(x, y, p=2)

该OP用于计算 `(x-y)` 的 p 范数（p-norm），需要注意这不是严格意义上的范数，仅作为距离的度量。输入 `x` 和 `y` 的形状（shape）必须是可广播的（broadcastable）。其含义如下，详情请参考 `numpy的广播概念 <https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html>`_ ：

- 每个输入都至少有1维
- 对两个输入的维度从后向前匹配，两个输入每一维的大小需要满足3个条件中的任意一个：相等、其中一个为1或者其中一个不存在。

定义 `z = x - y` ，`x` 和 `y` 的形状是可广播的，那么 `z` 的形状可按照下列步骤得到：

(1) 如果 `x` 和 `y` 的维数不同，先对维数较少的这个输入的维度往前补1。

例如，`x` 的形状为[8, 1, 6, 1]，`y` 的形状为[7, 1, 5]，对 `y` 的维度补1，

x (4-D Tensor):  8 x 1 x 6 x 1

y (4-D Tensor):  1 x 7 x 1 x 5

(2) 确定输出 `z` 每一维度的大小：从两个输入的维度中选取最大值。

z (4-D Tensor):  8 x 7 x 6 x 5

若两个输入的维数相同，则输出的大小可直接用步骤2确定。以下是 `p` 取不同值时，范数的计算公式：

当 `p = 0` ，定义 $0^0 = 0$，则 z 的零范数是 `z` 中非零元素的个数。

.. math::
    ||z||_{0}=\lim_{p \rightarrow 0}\sum_{i=1}^{m}|z_i|^{p}

当 `p = inf` ，`z` 的无穷范数是 `z` 所有元素中的最大值。

.. math::
    ||z||_\infty=\max_i |z_i|

当 `p = -inf` ，`z` 的负无穷范数是 `z` 所有元素中的最小值。

.. math::
    ||z||_{-\infty}=\min_i |z_i|

其他情况下，`z` 的 `p` 范数使用以下公式计算：

.. math::
    ||z||_{p}=(\sum_{i=1}^{m}|z_i|^p)^{\frac{1}{p}}

参数:
  - **x** (Variable): 1-D 到 6-D Tensor，数据类型为float32或float64。
  - **y** (Variable): 1-D 到 6-D Tensor，数据类型为float32或float64。
  - **p** (float, optional): 用于设置需要计算的范数，数据类型为float32或float64。默认值为2.

返回: `(x-y)` 的 `p` 范数。

返回类型: Variable

**代码示例**:

..  code-block:: python

    import paddle
    import paddle.fluid as fluid
    import numpy as np
    with fluid.dygraph.guard():
        x = fluid.dygraph.to_variable(np.array([[3, 3],[3, 3]]).astype(np.float32))
        y = fluid.dygraph.to_variable(np.array([[3, 3],[3, 1]]).astype(np.float32))
        out = paddle.dist(x, y, 0)
        print(out.numpy()) # out = [1.]
        out = paddle.dist(x, y, 2)
        print(out.numpy()) # out = [2.]
        out = paddle.dist(x, y, float("inf"))
        print(out.numpy()) # out = [2.]
        out = paddle.dist(x, y, float("-inf"))
        print(out.numpy()) # out = [0.]
