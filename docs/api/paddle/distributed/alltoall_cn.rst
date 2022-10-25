.. _cn_api_distributed_alltoall:

alltoall
-------------------------------


.. py:function:: paddle.distributed.alltoall(in_tensor_list, out_tensor_list, group=None, use_calc_stream=True)

将 in_tensor_list 里面的 tensors 按照卡数均分并按照卡的顺序分发到所有参与的卡并将结果 tensors 汇总到 out_tensor_list。
如下图所示，GPU0 卡的 in_tensor_list 会按照两张卡拆分成 0_0 和 0_1， GPU1 卡的 in_tensor_list 同样拆分成 1_0 和 1_1，经过 alltoall 算子后，
GPU0 卡的 0_0 会发送给 GPU0，GPU0 卡的 0_1 会发送给 GPU1，GPU1 卡的 1_0 会发送给 GPU0，GPU1 卡的 1_1 会发送给 GPU1，所以 GPU0 卡的 out_tensor_list 包含 0_0 和 1_0，
GPU1 卡的 out_tensor_list 包含 0_1 和 1_1。

.. image:: ./img/alltoall.png
  :width: 800
  :alt: alltoall
  :align: center

参数
:::::::::
    - **in_tensor_list** (list) - 包含所有输入 Tensors 的一个列表。在列表里面的所有元素都必须是一个 Tensor，Tensor 的数据类型必须是 float16、float32、 float64、int32、int64。
    - **out_tensor_list** (Tensor) - 包含所有输出 Tensors 的一个列表。在列表里面的所有元素数据类型要和输入的 Tensors 数据类型一致。
    - **group** (Group，可选) - new_group 返回的 Group 实例，或者设置为 None 表示默认地全局组。默认值：None。
    - **use_calc_stream** (bool，可选) - 标识使用计算流还是通信流。默认值：True。

返回
:::::::::
无

代码示例
:::::::::
COPY-FROM: paddle.distributed.alltoall
