.. _cn_api_linalg_svd:

svd
-------------------------------

.. py:function:: paddle.linalg.svd(x, full_matrics=False, name=None)


计算一个或一批矩阵的奇异值分解。

记 :math:`X` 为一个矩阵，则计算的结果为2个矩阵 :math:`U`, :math:`VH` 和一个向量 :math:`S` 。则分解后满足公式：

.. math::
    X = U * diag(S) * VH

值得注意的是，:math:`S` 是向量，从大到小表示每个奇异值。而 :math:`VH` 则是V的共轭转置。


参数：
:::::::::
    - **x** (Tensor) : 输入的欲进行奇异值分解的一个或一批方阵， 类型为 Tensor。 ``x`` 的形状应为 ``[*, M, N]``， 其中 ``*`` 为零或更大的批次维度， 数据类型支持float32， float64。
    - **full_matrics** (bool) : 是否计算完整的U和V矩阵， 类型为 bool 默认为 False。 这个参数会影响U和V生成的Shape。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回：
:::::::::
    - Tensor U， 奇异值分解的U矩阵。如果full_matrics设置为False，则Shape为 ``[*, M, K]`` ，如果full_metrics设置为True，那么Shape为 ``[*, M, M]`` 。其中K为M和N的最小值。
    - Tensor S， 奇异值向量，Shape为 ``[*, K]`` 。
    - Tensor VH， 奇异值分解的VH矩阵。如果full_matrics设置为False，则Shape为 ``[*, K, N]`` ，如果full_metrics设置为True，那么Shape为 ``[*, N, N]`` 。其中K为M和N的最小值。

代码示例：
::::::::::

.. code-block:: python

    import paddle

    x = paddle.to_tensor([[1.0, 2.0], [1.0, 3.0], [4.0, 6.0]]).astype('float64')
    x = x.reshape([3, 2])
    u, s, vh = paddle.linalg.svd(x)
    print (u)
    #U = [[ 0.27364809, -0.21695147  ],
    #      [ 0.37892198, -0.87112408 ],
    #      [ 0.8840446 ,  0.44053933 ]]

    print (s)
    #S = [8.14753743, 0.78589688]
    print (vh)
    #VT= [[ 0.51411221,  0.85772294],
    #     [ 0.85772294, -0.51411221]]
    
    # one can verify : U * S * VT == X
    #                  U * UH == I 
    #                  V * VH == I
