.. _cn_api_paddle_meshgrid:

meshgrid
-------------------------------

.. py:function:: paddle.meshgrid(*args, name=None, indexing=None)




对每个 Tensor 做扩充操作。输入是 Tensor 或者包含 Tensor 的列表，包含 k 个一维 Tensor，输出 k 个 k 维 Tensor。别名：``paddle.functional.meshgrid``。

参数
::::::::::::

         - **args** (Tensor|Tensor 数组) - 输入变量为 k 个一维 Tensor，形状分别为(N1,), (N2,), ..., (Nk, )。支持数据类型为 bfloat16、float16、float32、float64、uint16、int32、int64、complex64 和 complex128。
         - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。
         - **indexing** (str，可选) - 索引模式，可选 ``"xy"`` 或 ``"ij"``。默认值为 ``"ij"``。当选择 ``"xy"`` 时，前两个维度的顺序与前两个输入的基数相反；当选择 ``"ij"`` 时，各维度顺序与输入顺序一致。



返回
::::::::::::

k 个 k 维 ``Tensor``，每个形状均为(N1, N2, ..., Nk)。


代码示例
::::::::::::



COPY-FROM: paddle.meshgrid
