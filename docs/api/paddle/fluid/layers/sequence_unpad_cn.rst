.. _cn_api_fluid_layers_sequence_unpad:

sequence_unpad
-------------------------------


.. py:function:: paddle.fluid.layers.sequence_unpad(x, length, name=None)




.. note::
    该OP的输入为Tensor，输出为LoDTensor。该OP用于移除填充元素，与之对应，还存在进行数据填充的OP sequence_pad，详情见：:ref:`cn_api_fluid_layers_sequence_pad`

该OP根据length的信息，将input中padding（填充）元素移除，并且返回一个LoDTensor。

::

    示例：

    给定输入变量 ``x`` :
        x.data = [[ 1.0,  2.0,  3.0,  4.0,  5.0],
                  [ 6.0,  7.0,  8.0,  9.0, 10.0],
                  [11.0, 12.0, 13.0, 14.0, 15.0]],

    其中包含 3 个被填充到长度为5的序列，实际长度由输入变量 ``length`` 指明，其中，x的维度为[3,4]，length维度为[3]，length的第0维与x的第0维一致：

        length.data = [2, 3, 4],

    则去填充（unpad）后的输出变量为：

        out.data = [[1.0, 2.0, 6.0, 7.0, 8.0, 11.0, 12.0, 13.0, 14.0]]
        out.lod = [[0, 2, 5, 9]]



参数
::::::::::::

  - **x** (Variable) – 包含填充元素的Tensor，其维度大小不能小于2，支持的数据类型：float32, float64,int32, int64。
  - **length** (Variable) – 存储每个样本实际长度信息的1D Tesnor，该Tensor维度的第0维必须与x维度的第0维一致。支持的数据类型：int64。
  - **name**  (str，可选) – 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为None。

返回
::::::::::::
将输入的填充元素移除，并返回一个LoDTensor，其递归序列长度与length参数的信息一致，其数据类型和输入一致。

返回类型
::::::::::::
Variable

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.sequence_unpad