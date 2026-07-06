.. _cn_api_paddle_linalg_eigh:

eigh
-------------------------------

.. py:function:: paddle.linalg.eigh(x, UPLO='L', name=None, *, out=None)
计算厄米特矩阵或者实数对称矩阵的特征值和特征向量。

参数
::::::::::::

    - **x** (Tensor)：输入一个或一批厄米特矩阵或者实数对称矩阵。``x`` 的形状应为 ``[*, M, M]``，其中 ``*`` 为零或更大的批次维度，数据类型支持 float32、float64、complex64、complex128。别名 ``input``。
    - **UPLO** (str，可选)：表示计算上三角或者下三角矩阵，默认值为 'L'，表示计算下三角矩阵的特征值和特征向量，'U'表示计算上三角矩阵的特征值和特征向量。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

关键字参数
::::::::::::
    - **out** (tuple[Tensor, Tensor]，可选) - 输出 Tensor 元组，若不为 ``None``，计算结果将保存在该 Tensor 元组中，默认值为 ``None``。

返回
::::::::::::

    - Tensor out_value，输出矩阵的特征值，输出顺序按照从小到大进行排序。Shape 为 ``[*, M]`` 。
    - Tensor out_vector，输出矩阵的特征向量，与特征值一一对应，Shape 为 ``[*, M, M]`` 。

代码示例
::::::::::

COPY-FROM: paddle.linalg.eigh
