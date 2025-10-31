.. _cn_api_paddle_nn_ReplicationPad1d:

ReplicationPad1d
-------------------------------
.. py:class:: paddle.nn.ReplicationPad1d(padding, data_format="NCL", name=None)

通过复制边缘值来填充输入张量的边界。

参数
::::::::::::

  - **padding** (Tensor|Sequence[int]|int) - 填充大小。如果 `padding` 是一个整数，则在左、右两边都应用相同的填充。如果 `padding` 是一个包含两个整数的列表或元组，它将被解析为 `(pad_left, pad_right)`。
  - **data_format** (str，可选) - 指定输入的数据格式，可为 ``'NCL'`` 或 ``'NLC'``。默认值：``'NCL'``。
  - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
Tensor，填充后的张量。

代码示例
::::::::::::

COPY-FROM: paddle.nn.ReplicationPad1d
