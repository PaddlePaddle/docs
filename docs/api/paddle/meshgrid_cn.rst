.. _cn_api_paddle_meshgrid:

meshgrid
-------------------------------

.. py:function:: paddle.meshgrid(*args, **kargs)




对每个 Tensor 做扩充操作。输入是 Tensor 或者包含 Tensor 的列表，包含 k 个一维 Tensor，输出 k 个 k 维 Tensor。

参数
::::::::::::

         - **args** (Tensor|Tensor 数组) - 输入变量为 k 个一维 Tensor，形状分别为(N1,), (N2,), ..., (Nk, )。支持数据类型为 float32、float64、int32 和 int64。
         - **kargs** (可选) - 目前只接受 name 参数（str），具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。



返回
::::::::::::

k 个 k 维 ``Tensor``，每个形状均为(N1, N2, ..., Nk)。


代码示例
::::::::::::



COPY-FROM: paddle.meshgrid
