.. _cn_api_linalg_qr:

qr
-------------------------------

.. py:function:: paddle.linalg.qr(x, mode="reduced", name=None)


计算一个或一批矩阵的正交三角分解，也称QR分解（暂不支持反向）。

记 :math:`X` 为一个矩阵，则计算的结果为2个矩阵 :math:`Q` 和 :math:`R` ，则满足公式：

.. math::
    X = Q * R 

其中，:math:`Q` 是正交矩阵，:math:`R` 是上三角矩阵。


参数：
:::::::::
    - **x** (Tensor) : 输入进行正交三角分解的一个或一批方阵， 类型为 Tensor。 ``x`` 的形状应为 ``[*, M, N]``， 其中 ``*`` 为零或更大的批次维度， 数据类型支持float32， float64。
    - **mode** (str, 可选) : 控制正交三角分解的行为，默认是 ``reduced`` ，假设 ``x`` 形状应为 ``[*, M, N]`` 和 ``K = min(M, N)``：如果 ``mode = "reduced"`` ，则 :math:`Q` 形状为 ``[*, M, K]`` 和 :math:`R` 形状为 ``[*, K, N]`` ; 如果 ``mode = "complete"`` ，则 :math:`Q` 形状为 ``[*, M, M]`` 和 :math:`R` 形状为 ``[*, M, N]`` ; 如果 ``mode = "r"`` ，则不返回 :math:`Q`, 只返回 :math:`R` 且形状为 ``[*, K, N]`` 。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回：
:::::::::
    - Tensor Q， 正交三角分解的Q正交矩阵，需注意如果 ``mode = "reduced"`` ，则不返回Q矩阵，只返回R矩阵。
    - Tensor R， 正交三角分解的R上三角矩阵。

代码示例：
::::::::::

.. code-block:: python

    import paddle 

    x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).astype('float64')
    q, r = paddle.linalg.qr(x)
    print (q)
    print (r)

    # Q = [[-0.16903085,  0.89708523],
    #      [-0.50709255,  0.27602622],
    #      [-0.84515425, -0.34503278]])

    # R = [[-5.91607978, -7.43735744],
    #      [ 0.        ,  0.82807867]])
    
    # one can verify : X = Q * R ; 
