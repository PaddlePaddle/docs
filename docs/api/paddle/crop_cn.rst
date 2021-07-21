.. _cn_api_crop:

crop
-------------------------------

.. py:function:: paddle.crop(x, shape=None, offsets=None, name=None)




根据偏移量（offsets）和形状（shape），裁剪输入（x）Tensor。

**示例**：

::

    * 示例1（输入为2-D Tensor）：

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

    * 示例2（输入为3-D Tensor）：

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

参数:
  - **x** (Tensor): 1-D到6-D Tensor，数据类型为float32、float64、int32或者int64。
  - **shape** (list|tuple|Tensor) - 输出Tensor的形状，数据类型为int32。如果是列表或元组，则其长度必须与x的维度大小相同，如果是Tensor，则其应该是1-D Tensor。当它是列表时，每一个元素可以是整数或者形状为[1]的Tensor。含有Variable的方式适用于每次迭代时需要改变输出形状的情况。
  - **offsets** (list|tuple|Tensor，可选) - 每个维度上裁剪的偏移量，数据类型为int32。如果是列表或元组，则其长度必须与x的维度大小相同，如果是Tensor，则其应是1-D Tensor。当它是列表时，每一个元素可以是整数或者形状为[1]的Variable。含有Variable的方式适用于每次迭代的偏移量（offset）都可能改变的情况。默认值：None，每个维度的偏移量为0。
  - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回: 裁剪后的Tensor，数据类型与输入（x）相同。

返回类型: Tensor

**代码示例**:
COPY-FROM: paddle.crop:code-example1
COPY-FROM: paddle.crop
    
    

