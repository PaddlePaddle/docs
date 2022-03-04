.. _cn_api_linalg_lu_unpack:

lu_unpack
-------------------------------

.. py:function:: paddle.linalg.lu_unpack(x, y, unpack_ludata=True, unpack_pivots=True, name=None)

对paddle.linalg.lu返回结果的LU、pivot进行展开得到原始的单独矩阵L、U、P。

从LU中获得下三角矩阵L, 上三角矩阵U。
从序列pivot转换得到矩阵P，其转换过程原理如下伪代码所示：

.. code-block:: text

    ones = eye(rows) #eye matrix of rank rows
    for i in range(cols):
        swap(ones[i], ones[pivots[i]])
    return ones

参数：
:::::::::
    - **x** (Tensor) - paddle.linalg.lu返回结果的LU矩阵。
    - **y** (Tensor) - paddle.linalg.lu返回结果的pivot序列。
    - **unpack_ludata** (bool, 可选) - 若为True，则对输入x(LU)进行展开得到L、U，否则。默认True。
    - **unpack_pivots** (bool, 可选) - 若为True，则对输入y(pivots)序列进行展开，得到转换矩阵P。默认True。
    - **name** (str, 可选) - 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回：
:::::::::
    - Tensor L, 由LU展开得到的L矩阵，若unpack_ludata为False，则为None。
    - Tensor U, 由LU展开得到的U矩阵，若unpack_ludata为False，则为None。
    - Tensor P, 由序列pivots展开得到的旋转矩阵P，若unpack_pivots为False，则为None。

代码示例：
::::::::::

.. code-block:: python

    import paddle 

    x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).astype('float64')
    lu,p,info = paddle.linalg.lu(x, get_infos=True)

    # >>> lu:
    # Tensor(shape=[3, 2], dtype=float64, place=CUDAPlace(0), stop_gradient=True,
    #    [[5.        , 6.        ],
    #        [0.20000000, 0.80000000],
    #        [0.60000000, 0.50000000]])
    # >>> p
    # Tensor(shape=[2], dtype=int32, place=CUDAPlace(0), stop_gradient=True,
    #    [3, 3])
    # >>> info
    # Tensor(shape=[], dtype=int32, place=CUDAPlace(0), stop_gradient=True,
    #    0)
    
    P,L,U = paddle.linalg.lu_unpack(lu,p)

    # >>> P
    # (Tensor(shape=[3, 3], dtype=float64, place=CUDAPlace(0), stop_gradient=True,
    # [[0., 1., 0.],
    # [0., 0., 1.],
    # [1., 0., 0.]]), 
    # >>> L
    # Tensor(shape=[3, 2], dtype=float64, place=CUDAPlace(0), stop_gradient=True,
    # [[1.        , 0.        ],
    # [0.20000000, 1.        ],
    # [0.60000000, 0.50000000]]), 
    # >>> U
    # Tensor(shape=[2, 2], dtype=float64, place=CUDAPlace(0), stop_gradient=True,
    # [[5.        , 6.        ],
    # [0.        , 0.80000000]]))
    

    # one can verify : X = P @ L @ U ;  
    