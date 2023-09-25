.. _cn_api_paddle_static_nn_sequence_unpad:

sequence_unpad
-------------------------------

.. py:function:: paddle.static.nn.sequence_unpad(x, length, name=None)




.. note::
    该 API 的输入为 Tensor，输出为带有 LoD 信息的 Tensor。用于移除填充元素，与之对应，还存在进行数据填充的 API :ref:`cn_api_paddle_static_nn_sequence_pad`。

根据 length 的信息，将 input 中 padding（填充）元素移除，并且返回一个带有 LoD 信息的 Tensor。

::

    示例：

    给定输入变量 ``x`` :
        x.data = [[ 1.0,  2.0,  3.0,  4.0,  5.0],
                  [ 6.0,  7.0,  8.0,  9.0, 10.0],
                  [11.0, 12.0, 13.0, 14.0, 15.0]],

    其中包含 3 个被填充到长度为 5 的序列，实际长度由输入变量 ``length`` 指明，其中，x 的维度为[3,4]，length 维度为[3]，length 的第 0 维与 x 的第 0 维一致：

        length.data = [2, 3, 4],

    则去填充（unpad）后的输出变量为：

        out.data = [[1.0, 2.0, 6.0, 7.0, 8.0, 11.0, 12.0, 13.0, 14.0]]
        out.lod = [[0, 2, 5, 9]]



参数
:::::::::
  - **x** (Tensor) – 包含填充元素的 Tensor，其维度大小不能小于 2，支持的数据类型：float32, float64,int32, int64。
  - **length** (Tensor) – 存储每个样本实际长度信息的 1D Tesnor，该 Tensor 维度的第 0 维必须与 x 维度的第 0 维一致。支持的数据类型：int64。
  - **name**  (str，可选) – 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
将输入的填充元素移除，并返回一个带有 LoD 信息的 Tensor，其递归序列长度与 length 参数的信息一致，其数据类型和输入一致。

代码示例
:::::::::
COPY-FROM: paddle.static.nn.sequence_unpad
