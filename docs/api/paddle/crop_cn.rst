.. _cn_api_paddle_crop:

crop
-------------------------------

.. py:function:: paddle.crop(x, shape=None, offsets=None, name=None)




根据偏移量（offsets）和形状（shape），裁剪输入（x）Tensor。

**示例**：

::

    * 示例 1（输入为 2-D Tensor）：

        输入：
            X.shape = [3, 5]
            X.data = [[0, 1, 2, 0, 0],
                      [0, 3, 4, 0, 0],
                      [0, 0, 0, 0, 0]]

        参数：
            shape = [2, 2]
            offsets = [0, 1]

        输出：
            Out.shape = [2, 2]
            Out.data = [[1, 2],
                        [3, 4]]

    * 示例 2（输入为 3-D Tensor）：

        输入：

            X.shape = [2, 3, 4]
            X.data =  [[[0, 1, 2, 3],
                        [0, 5, 6, 7],
                        [0, 0, 0, 0]],
                       [[0, 3, 4, 5],
                        [0, 6, 7, 8],
                        [0, 0, 0, 0]]]

        参数：
            shape = [2, 2, -1]
            offsets = [0, 0, 1]

        输出：
            Out.shape = [2, 2, 3]
            Out.data = [[[1, 2, 3],
                         [5, 6, 7]],
                        [[3, 4, 5],
                         [6, 7, 8]]]

**示例二图解说明**：

    下图展示了示例二中的情形——一个形状为[2,2,2]的三维张量通过 crop 操作裁剪为形状为[1,2,2]的三维张量，同时保持了张量中元素的顺序和值不变。通过比较，可以清晰地看到张量形状变化前后各元素的对应关系。

    .. figure:: ../../images/api_legend/crop.png
       :width: 500
       :alt: 示例二图示
       :align: center

参数
:::::::::

  - **x** (Tensor) - 1-D 到 6-D Tensor，数据类型为 float32、float64、int32 或者 int64。
  - **shape** (list|tuple|Tensor，可选) - 输出 Tensor 的形状，数据类型为 int32。如果是列表或元组，则其长度必须与 x 的维度大小相同，如果是 Tensor，则其应该是 1-D Tensor。当它是列表时，每一个元素可以是整数或者形状为[]的 0-D Tensor。含有 Tensor 的方式适用于每次迭代时需要改变输出形状的情况。
  - **offsets** (list|tuple|Tensor，可选) - 每个维度上裁剪的偏移量，数据类型为 int32。如果是列表或元组，则其长度必须与 x 的维度大小相同，如果是 Tensor，则其应是 1-D Tensor。当它是列表时，每一个元素可以是整数或者形状为[]的 0-D Tensor。含有 Tensor 的方式适用于每次迭代的偏移量（offset）都可能改变的情况。默认值：None，每个维度的偏移量为 0。
  - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
裁剪后的 Tensor，数据类型与输入（x）相同。



代码示例
:::::::::

COPY-FROM: paddle.crop
