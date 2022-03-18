.. _cn_api_linalg_corrcoef:

corrcoef
-------------------------------

.. py:function:: paddle.linalg.corrcoef(x, rowvar=True, ddof=False, name=None)


给定输入Tensor，计算输入Tensor的皮尔逊积矩相关系数矩阵。
细节请参考cov文档. 皮尔逊积矩相关系数 `R`和 协方差矩阵`C`的关系如下：

    .. math:: R_{ij} = \\frac{ C_{ij} } { \\sqrt{ C_{ii} * C_{jj} } }

    `R`的值在-1到1之间.

参数：
:::::::::
    - **x** (Tensor) - 一个N(N<=2)维矩阵，包含多个变量。默认矩阵的每行是一个观测变量，由参数rowvar设置。
    - **rowvar** (bool, 可选) - 若是True，则每行作为一个观测变量；若是False，则每列作为一个观测变量。默认True。
    - **ddof** (bool, 可选) - 在计算中不起作用，不需要。默认False。
    - **name** (str, 可选) - 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回：
:::::::::
    - Tensor, 输入x的皮尔逊积矩相关系数矩阵。

代码示例：
::::::::::

.. code-block:: python

    import paddle

    xt = paddle.rand((3,4))
    paddle.linalg.corrcoef(xt)

    '''
    Tensor(shape=[3, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
        [[ 1.        , -0.73702252,  0.66228950],
        [-0.73702258,  1.        , -0.77104872],
        [ 0.66228974, -0.77104825,  1.        ]])
    '''
    
