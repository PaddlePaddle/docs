.. _cn_api_linalg_corrcoef:

corrcoef
-------------------------------

.. py:function:: paddle.linalg.corrcoef(x, rowvar=True, name=None)


给定输入Tensor，计算输入Tensor的皮尔逊积矩相关系数矩阵。
细节请参考 `cov文档 https://github.com/PaddlePaddle/docs/blob/develop/docs/api/paddle/linalg/cov_cn.rst` 。
皮尔逊积矩相关系数 `R` 和协方差矩阵 `C` 的关系如下：

    .. math:: R_{ij} = \\frac{ C_{ij} } { \\sqrt{ C_{ii} * C_{jj} } }

    `R` 的值在-1到1之间。

参数
:::::::::
    - **x** (Tensor) - 一个N(N<=2)维矩阵，包含多个变量。默认矩阵的每行是一个观测变量，由参数rowvar设置。
    - **rowvar** (bool，可选) - 若是True，则每行作为一个观测变量；若是False，则每列作为一个观测变量。默认True。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为None。

返回
:::::::::
    - Tensor，输入x的皮尔逊积矩相关系数矩阵。

代码示例
::::::::::
COPY-FROM: <paddle.linalg.corrcoef>:<code-example1>
