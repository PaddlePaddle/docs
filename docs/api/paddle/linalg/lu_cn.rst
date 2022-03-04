.. _cn_api_linalg_lu:

lu
-------------------------------

.. py:function:: paddle.linalg.lu(x, pivot=True, get_infos=False, name=None)

对输入的N维(N>=2)矩阵x进行LU分解。

返回LU分解矩阵L、U和旋转矩阵P。L是下三角矩阵，U是上三角矩阵，拼接成单个矩阵LU，函数直接返回LU。

如果pivot为True则返回旋转矩阵P对应序列pivot，序列pivot转换到矩阵P可以经如下伪代码实现：

.. code-block:: text

    ones = eye(rows) #eye matrix of rank rows
    for i in range(cols):
        swap(ones[i], ones[pivots[i]])
    return ones

.. note::

    pivot选项只在gpu下起作用, cpu下暂不支持为False，会报错。

LU和pivot可以通过调用paddle.linalg.lu_unpack展开获得L、U、P矩阵。

参数：
:::::::::
    - **x** (Tensor) - 需要进行LU分解的输入矩阵x，x是维度大于2维的矩阵。
    - **pivot** (bool, 可选) - LU分解时是否进行旋转。若为True则执行旋转操作，若为False则不执行旋转操作，该选项只在gpu下起作用, cpu下暂不支持为False，会报错。默认True。
    - **get_infos** (bool, 可选) - 是否返回分解状态信息，若为True，则返回分解状态Tensor，否则不返回。默认False。
    - **name** (str, 可选) - 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回：
:::::::::
    - Tensor LU, LU分解结果矩阵LU，由L、U拼接组成。
    - Tensor(dtype=int) Pivots, 旋转矩阵对应的旋转序列，详情见说明部分pivot部分, 对于输入[*,m,n]的x，Pivots shape为[*, m]。
    - Tensor(dtype=int) Infos, 矩阵分解状态信息矩阵，对于输入[*,m,n], Infos shape为[*]。每个元素表示每组矩阵的LU分解是否成功，0表示分解成功。

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
