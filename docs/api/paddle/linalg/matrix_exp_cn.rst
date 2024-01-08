.. _cn_api_paddle_linalg_matrix_exp:

matrix_exp
-------------------------------

.. py:function:: paddle.linalg.matrix_exp(x, name=None)

计算方阵的矩阵指数。

.. math::

    exp(A) = \sum_{n=0}^\infty A^n/n!

输入的张量 ``x`` 应该是方阵，比如 ``(*, M, M)`` ，矩阵指数通过比例平方法的帕德近似计算而来。

[1] Nicholas J. Higham, The scaling and squaring method for the matrix exponential revisited.


参数
::::::::::::

    - **x** (Tensor) - 输入张量，形状应该为 ``(*, M, M)`` ， ``*`` 表示 0 或多个维度。数据类型为： ``float32``, ``float64`` 。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
Tensor，与 ``x`` 的类型和形状相同。

代码示例
::::::::::

COPY-FROM: paddle.linalg.matrix_exp
