.. _cn_api_tensor_cond:

cond
-------------------------------

.. py:function:: paddle.cond(x, p=None, name=None)


根据范数种类 ``p`` 计算一个或一批矩阵的条件数。

参数
:::::::::
    - **x** (Tensor) : 输入可以是形状为``(*, m, n)``的矩阵Tensor，``*``为零或更大的批次维度，此时``p``为`2`或`-2`；也可以是形状为``(*, n, n)``的可逆（批）方阵Tensor，此时``p``为任意已支持的值。数据类型为 float32 或 float64。
    - **p** (float|string，可选) : 范数种类。目前支持的值为 `fro`、`nuc`、`1`、`-1`、`2`、`-2`、`inf`、`-inf`。默认值为 `None` ，即范数种类为 `2` 。
    - **name** (str，可选) - 默认值为 `None` ，一般无需设置，具体用法请参见 :ref:`api_guide_Name` 。

返回：
:::::::::
    - Tensor， 条件数的计算结果， 数据类型和输入``x``的一致。

代码示例
::::::::::

.. code-block:: python

    import paddle
    import numpy as np

    np_input = [[1., 0, -1], [0, 1, 0], [1, 0, 1]]
    x = paddle.to_tensor(np_input)
    # [[1., 0, -1], [0, 1, 0], [1, 0, 1]]

    # compute conditional number when p is None
    out = paddle.linalg.cond(x)
    # out.numpy() [1.4142135]

    # compute conditional number when order of the norm is 'fro'
    out_fro = paddle.linalg.cond(x, p='fro')
    # out_fro.numpy() [3.1622777]

    # compute conditional number when order of the norm is 'nuc'
    out_nuc = paddle.linalg.cond(x, p='nuc')
    # out_nuc.numpy() [9.2426405]

    # compute conditional number when order of the norm is 1
    out_1 = paddle.linalg.cond(x, p=1)
    # out_1.numpy() [2.]

    # compute conditional number when order of the norm is -1
    out_minus_1 = paddle.linalg.cond(x, p=-1)
    # out_minus_1.numpy() [1.]

    # compute conditional number when order of the norm is 2
    out_2 = paddle.linalg.cond(x, p=2)
    # out_2.numpy() [1.4142135]

    # compute conditional number when order of the norm is -1
    out_minus_2 = paddle.linalg.cond(x, p=-2)
    # out_minus_2.numpy() [0.70710677]

    # compute conditional number when order of the norm is inf
    out_inf = paddle.linalg.cond(x, p=np.inf)
    # out_inf.numpy() [2.]

    # compute conditional number when order of the norm is -inf
    out_minus_inf = paddle.linalg.cond(x, p=-np.inf)
    # out_minus_inf.numpy() [1.]

    a = paddle.to_tensor(np.random.randn(2, 4, 4).astype('float32'))
    # a.numpy() 
    # [[[ 0.14063153 -0.996288    0.7996131  -0.02571543]
    #   [-0.16303636  1.5534962  -0.49919784 -0.04402903]
    #   [-1.1341571  -0.6022629   0.5445269   0.29154757]
    #   [-0.16816919 -0.30972657  1.7521842  -0.5402487 ]]
    #  [[-0.58081484  0.12402827  0.7229862  -0.55046535]
    #   [-0.15178485 -1.1604939   0.75810957  0.30971205]
    #   [-0.9669573   1.0940945  -0.27363303 -0.35416734]
    #   [-1.216529    2.0018666  -0.7773689  -0.17556527]]]
    a_cond_fro = paddle.linalg.cond(a, p='fro')
    # a_cond_fro.numpy()  [31.572273 28.120834]

    b = paddle.to_tensor(np.random.randn(2, 3, 4).astype('float64'))
    # b.numpy()
    # [[[ 1.61707487  0.46829144  0.38130416  0.82546736]
    #   [-1.72710298  0.08866375 -0.62518804  0.16128892]
    #   [-0.02822879 -1.67764516  0.11141444  0.3220113 ]]
    #  [[ 0.22524372  0.62474921 -0.85503233 -1.03960523]
    #   [-0.76620689  0.56673047  0.85064753 -0.45158196]
    #   [ 1.47595418  2.23646462  1.5701758   0.10497519]]]
    b_cond_2 = paddle.linalg.cond(b, p=2)
    # b_cond_2.numpy()  [3.30064451 2.51976252]