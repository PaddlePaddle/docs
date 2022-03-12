.. _cn_api_linalg_corrcoef:

corrcoef
-------------------------------

.. py:function:: paddle.linalg.corrcoef(x, rowvar=True, ddof=False, name=None)


给定输入Tensor，计算输入Tensor的皮尔逊积矩相关系数矩阵。

皮尔逊积矩相关系数矩阵是一个方阵，用于指示每两个输入元素之间的皮尔逊积矩相关系数。
例如对于有N个元素的输入X=[x1,x2,…xN]T，皮尔逊积矩相关系数矩阵的元素Cij表示输入xi和xj之间的皮尔逊积矩相关系数，Cii表示xi其自身的皮尔逊积矩相关系数。

参数：
:::::::::
    - **x** (Tensor) - 一个N(N<=2)维矩阵，包含多个变量。默认矩阵的每行是一个观测变量，由参数rowvar设置。
    - **rowvar** (bool, 可选) - 若是True，则每行作为一个观测变量；若是False，则每列作为一个观测变量。默认True。
    - **ddof** (bool, 可选) - 在计算中不起作用，不需要。默认False。

返回：
:::::::::
    - Tensor, 输入x的皮尔逊积矩相关系数矩阵。假设x是[m,n]的矩阵，rowvar=True, 则输出为[m,m]的矩阵。

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
    