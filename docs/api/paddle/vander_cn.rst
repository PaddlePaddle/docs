.. cn_api_tensor_vander:

paddle.vander
-------------------------------

.. py:function:: paddle.vander(x, N=None, increasing=False, name=None)
生成范德蒙德矩阵。

参数
::::::::::
    - **x** (Tensor) - 输入的一维 Tensor，支持的数据类型：int32、int64、float32、float64、complex64、complex128。
    - **N** (int) - 输出中的列数。如果未指定 N，则返回一个方阵(N = len(x))。
    - **increasing** (bool) - 列的幂次顺序。如果为 True，则幂次从左到右增加，如果为 False（默认值），则幂次顺序相反。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::
返回一个根据 N 和 increasing 创建的范德蒙德矩阵。

代码示例
::::::::::

COPY-FROM: paddle.vander
