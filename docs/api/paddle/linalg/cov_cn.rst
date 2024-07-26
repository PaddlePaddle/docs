.. _cn_api_paddle_linalg_cov:

cov
-------------------------------

.. py:function:: paddle.linalg.cov(x, rowvar=True, ddof=True, fweights=None, aweights=None, name=None)


给定输入 Tensor 和权重，计算输入 Tensor 的协方差矩阵。

协方差矩阵是一个方阵，用于指示每两个输入元素之间的协方差值。
例如对于有 N 个元素的输入 X=[x1,x2,…xN]T，协方差矩阵的元素 Cij 表示输入 xi 和 xj 之间的协方差，Cii 表示 xi 其自身的协方差。

参数
::::::::::::

    - **x** (Tensor) - 一个 N(N<=2)维矩阵，包含多个变量。默认矩阵的每行是一个观测变量，由参数 rowvar 设置。
    - **rowvar** (bool，可选) - 若是 True，则每行作为一个观测变量；若是 False，则每列作为一个观测变量。默认 True。
    - **ddof** (bool，可选) - 若是 True，返回无偏估计结果；若是 False，返回普通平均值计算结果。默认 True。
    - **fweights** (Tensor，可选) - 包含整数频率权重的 1 维 Tensor，表示每一个观测向量的重复次数。其维度值应该与输入 x 的观测维度值相等，为 None 则不起作用，默认 None。
    - **aweights** (Tensor，可选) - 包含整数观测权重的 1 维 Tensor，表示每一个观测向量的重要性，重要性越高对应值越大。其维度值应该与输入 x 的观测维度值相等，为 None 则不起作用，默认 None。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::

Tensor，输入 x 的协方差矩阵。假设 x 是[m,n]的矩阵，rowvar=True，则输出为[m,m]的矩阵。

代码示例
::::::::::

COPY-FROM: paddle.linalg.cov
