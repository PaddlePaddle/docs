.. _cn_api_linalg_corrcoef:

corrcoef
-------------------------------

.. py:function:: paddle.linalg.corrcoef(x, rowvar=True, name=None)


相关系数矩阵表示输入矩阵中每对变量的相关性。例如，对于 N 维样本 X=[x1，x2，…xN]T，则相关系数矩阵
元素 `Rij` 是 `xi` 和 `xj` 的相关性。元素 `Rii` 是 `xi` 本身的协方差。

皮尔逊积矩相关系数 `R` 和协方差矩阵 `C` 的关系如下：

    .. math:: R_{ij} = \frac{ C_{ij} } { \sqrt{ C_{ii} * C_{jj} } }

    `R` 的值在-1 到 1 之间。

参数
:::::::::
    - **x** (Tensor) - 一个 N(N<=2)维矩阵，包含多个变量。默认矩阵的每行是一个观测变量，由参数 rowvar 设置。
    - **rowvar** (bool，可选) - 若是 True，则每行作为一个观测变量；若是 False，则每列作为一个观测变量。默认 True。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
    输入 x 的皮尔逊积矩相关系数矩阵。

代码示例
::::::::::

COPY-FROM: paddle.linalg.corrcoef
