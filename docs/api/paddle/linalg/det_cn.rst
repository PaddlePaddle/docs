.. _cn_api_linalg_eigh:

det
-------------------------------

.. py:function:: paddle.linalg.det(x)
计算批量矩阵的行列式值。

参数：
:::::::::
    - **x** (Tensor) : 输入一个或批量矩阵。 ``x`` 的形状应为 ``[*, M, M]``， 其中 ``*`` 为零或更大的批次维度， 数据类型支持float32， float64。

返回：
:::::::::
- Tensor out_value， 输出矩阵的行列式值 Shape为。 ``[*]`` 。

代码示例：
::::::::::

.. code-block:: python


    import paddle

    x =  paddle.randn([3,3,3])

    A = paddle.det(x)

    print(A)

    # [ 0.02547996,  2.52317095, -6.15900707])
