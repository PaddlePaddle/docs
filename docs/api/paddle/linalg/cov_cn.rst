.. _cn_api_linalg_cov:

cov
-------------------------------

.. py:function:: paddle.linalg.cov(x, rowvar=True, ddof=True, fweights=None, aweights=None, name=None)


给定输入Tensor和权重，计算输入Tensor的协方差矩阵。

协方差矩阵是一个方阵，用于指示每两个输入元素之间的协方差值。
例如对于有N个元素的输入X=[x1,x2,…xN]T，协方差矩阵的元素Cij表示输入xi和xj之间的协方差，Cij表示xi其自身的协方差。

参数：
:::::::::
    - **x** (Tensor) - 一个N(N<=2)维矩阵，包含多个变量。默认矩阵的每行是一个观测变量，由参数rowvar设置。
    - **rowvar** (bool, 可选) - 若是True，则每行作为一个观测变量；若是False，则每列作为一个观测变量。默认True。
    - **ddof** (bool, 可选) - 若是True，返回无偏估计结果；若是False，返回普通平均值计算结果。默认True。
    - **fweights** (Tensor, 可选) - 包含整数频率权重的1维Tensor，表示每一个观测向量的重复次数。其维度值应该与输入x的观测维度值相等，为None则不起作用，默认None。
    - **aweights** (Tensor, 可选) - 包含整数观测权重的1维Tensor，表示每一个观测向量的重要性，重要性越高对应值越大。其维度值应该与输入x的观测维度值相等，为None则不起作用，默认None。
    - **name** (str, 可选) - 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回：
:::::::::
    - Tensor, 输入x的协方差矩阵。假设x是[m,n]的矩阵，rowvar=True, 则输出为[m,m]的矩阵。

代码示例：
::::::::::

.. code-block:: python

    import paddle

    xt = paddle.rand((3,4))
    paddle.linalg.cov(xt)

    '''
    Tensor(shape=[3, 3], dtype=float64, place=CUDAPlace(0), stop_gradient=True,
        [[0.07918842, 0.06127326, 0.01493049],
            [0.06127326, 0.06166256, 0.00302668],
            [0.01493049, 0.00302668, 0.01632146]])
    '''
    