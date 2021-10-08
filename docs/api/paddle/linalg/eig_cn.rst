.. _cn_api_linalg_eig:

eig
-------------------------------

.. py:function:: paddle.linalg.eig(x, name=None)
计算一般方阵``x``的的特征值和特征向量。

.. note::
    - 如果输入矩阵 ``x`` 为Hermitian矩阵或实对称阵，请使用更快的API :ref:`paddle.linalg.eigh` 。
    - 如果只计算特征值，请使用 :ref:`paddle.linalg.eigvals` 。
    - 如果矩阵 ``x`` 不是方阵，请使用 :ref:`paddle.linalg.svd` 。
    - 该API当前只能在CPU上执行。
    - 对于输入是实数和复数类型，输出的数据类型均为复数。

参数：
:::::::::
    - **x** (Tensor) - 输入一个或一批矩阵。 ``x`` 的形状应为 ``[*, M, M]`` ， 数据类型支持float32，float64，complex64和complex128。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回：
:::::::::
    - Tensor Eigenvalues， 输出Shape为 ``[*, M]`` 的矩阵，表示特征值。
    - Tensor Eigenvectors， 输出Shape为 ``[*, M, M]`` 矩阵，表示特征向量。

代码示例：
::::::::::

.. code-block:: python

    import paddle

    paddle.device.set_device("cpu")

    x_data = paddle.to_tensor([[1.6707249, 7.2249975, 6.5045543],
                              [9.956216,  8.749598,  6.066444 ],
                              [4.4251957, 1.7983172, 0.370647 ]], dtype='float32')

    w, v = paddle.linalg.eig(x_data)
    print(v)
    # Tensor(shape=[3, 3], dtype=complex128, place=CPUPlace, stop_gradient=False,
    #       [[(-0.5061363550800655+0j) , (-0.7971760990842826+0j) ,
    #         (0.18518077798279986+0j)],
    #        [(-0.8308237755993192+0j) ,  (0.3463813401919749+0j) ,
    #         (-0.6837005269141947+0j) ],
    #        [(-0.23142567697893396+0j),  (0.4944999840400175+0j) ,
    #         (0.7058765252952796+0j) ]])

    print(w)
    # Tensor(shape=[3], dtype=complex128, place=CPUPlace, stop_gradient=False,
    #       [ (16.50471283351188+0j)  , (-5.5034820550763515+0j) ,
    #         (-0.21026087843552282+0j)])
