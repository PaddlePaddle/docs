.. _cn_api_paddle_vander:

vander
-------------------------------

.. py:function:: paddle.vander(x, n=None, increasing=False, name=None)
生成范德蒙德矩阵。

输出矩阵的每一列都是输入向量的幂。 幂的顺序由递增的布尔参数确定。 具体而言，当递增为 ``false`` 时，第 i 个输出列是输入向量元素顺序的升序，其幂为 N-i-1。 每行都有等比级数的这样一个矩阵称为 Alexandre-Theophile Vandermonde 矩阵。

参数
::::::::::
    - **x** (Tensor) - 输入的 Tensor，必须是 1-D Tensor, 支持的数据类型：int32、int64、float32、float64、complex64、complex128。
    - **n** (int，可选) - 输出中的列数。如果未指定 n，则返回一个方阵(n = len(x))。
    - **increasing** (bool，可选) - 列的幂次顺序。如果为 True，则幂次从左到右增加，如果为 False（默认值），则幂次顺序相反。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::
返回一个根据 n 和 increasing 创建的范德蒙德矩阵。

代码示例
::::::::::

COPY-FROM: paddle.vander
