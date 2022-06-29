.. _cn_api_distributed_alltoall_single:

alltoall_single
-------------------------------


.. py:function:: paddle.distributed.alltoall_single(in_tensor, out_tensor, in_split_sizes=None, out_split_sizes=None, group=None, use_calc_stream=True)

将 ``in_tensor`` 按照顺序和 ``in_split_sizes`` 拆分并分发到所有参与的卡上，并将结果按照 ``out_split_sizes``  聚合为 ``out_tensor`` 。

参数
:::::::::
    - in_tensor (Tensor) - 输入Tensor。数据类型必须是float16、float32、 float64、int32、int64。
    - out_tensor (Tensor) - 输出Tensor。数据类型要和 ``in_tensor`` 一致。
    - in_split_sizes (list[int], 可选) – 拆分操作依赖的size列表；如果是 None 或空，输出张量的第0维必须能被 world size 整除。
    - out_split_sizes (list[int], 可选) – 聚合操作依赖的size列表；如果是 None 或空，输出张量的第0维必须能被 world size 整除。
    - group (Group，可选) - new_group返回的Group实例，或者设置为None表示默认地全局组。默认值：None。
    - use_calc_stream (bool，可选) - 标识使用计算流还是通信流。默认值：True。

返回
:::::::::
如果 ``use_calc_stream`` 被设为False，则返回Task；反之则返回 None。

代码示例
:::::::::
COPY-FROM: paddle.distributed.alltoall_single