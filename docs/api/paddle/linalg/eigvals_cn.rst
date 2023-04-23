.. _cn_api_linalg_eigvals:

eigvals
-------------------------------

.. py:function:: paddle.linalg.eigvals(x, name=None)
计算一个（或一批）普通方阵的特征值。


.. note::
该 API 的反向实现尚未完成，若你的代码需要对其进行反向传播，请使用 ref:`cn_api_linalg_eig`。


参数
:::::::::

        - **x** （Tensor）- 需要计算特征值的方阵。输入的 Tensor 维度为 ``[*, M, M]``，其中 ``*`` 表示矩阵的批次维度。支持 ``float32`` 、 ``float64`` 、 ``complex64`` 和  ``complex128`` 四种数据类型。
        - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。


返回
:::::::::
``Tensor``，包含 x 的所有未排序特征值。返回的 Tensor 具有与 x 相同的批次维度。即使输入的 x 是实数 tensor，返回的也会是复数的结果。


代码示例
:::::::::
COPY-FROM: paddle.linalg.eigvals
