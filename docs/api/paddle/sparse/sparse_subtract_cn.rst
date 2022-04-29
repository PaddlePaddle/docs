.. _cn_api_sparse_subtract:

subtract
-------------------------------

.. py:function:: paddle.sparse.subtract(x, y, name=None)



逐元素相减算子，输入 ``x`` 与输入 ``y`` 逐元素相减，并将各个位置的输出元素保存到返回结果中。

输入 ``x`` 与输入 ``y`` 必须为相同形状且为相同稀疏压缩格式（同为 `SparseCooTensor` 或同为 `SparseCsrTensor` ），如果同为 `SparseCooTensor` 则 `sparse_dim` 也需要相同。

等式为：

.. math::
        Out = X - Y

- :math:`X` ：多维稀疏Tensor。
- :math:`Y` ：多维稀疏Tensor。

参数
:::::::::
    - x (Tensor) - 输入的Tensor，数据类型为：float32、float64。
    - y (Tensor) - 输入的Tensor，数据类型为：float32、float64。
    - name (str，可选） - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name` 。

返回
:::::::::
多维稀疏Tensor, 数据类型和压缩格式与 ``x`` 相同 。


代码示例
:::::::::

COPY-FROM: paddle.sparse.subtract