.. _cn_api_nn_functional_one_hot:

one_hot
-------------------------------

.. py:function:: paddle.nn.functional.one_hot(x, num_classes, name=None)
该OP将输入'x'中的每个id转换为一个one-hot向量，其长度为 ``num_classes`` ，该id对应的向量维度上的值为1，其余维度的值为0。

输出的Tensor的shape是在输入shape的最后一维后面添加了num_classes的维度。

- 示例1：

.. code-block:: text

  输入：
    X.shape = [4]
    X.data = [1, 1, 3, 0]
    num_classes = 4
  
  输出：
    Out.shape = [4, 4]
    Out.data = [[0., 1., 0., 0.],
                [0., 1., 0., 0.],
                [0., 0., 0., 1.],
                [1., 0., 0., 0.]]

- 示例2：

.. code-block:: text

  输入：
    X.shape = [4]
    X.data = [1, 1, 5, 0]
    num_classes = 4

  输出：抛出 Illegal value 的异常
    X中第2维的值是5，超过了num_classes，因此抛异常。


参数：
    - **x** (Tensor) - 维度为 :math:`[N_1, ..., N_n]` 的多维Tensor，维度至少1维。数据类型为int32或int64。
    - **num_classes** (int) - 用于定义一个one-hot向量的长度。若输入为词id，则 ``num_classes`` 通常取值为词典大小。
    - **name** (str，可选) – 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回：Tensor，转换后的one_hot Tensor，数据类型为float32。

**代码示例**：

.. code-block:: python

    import paddle
    label = paddle.to_tensor([1, 1, 3, 0], dtype='int64')
    # label.shape = [4]
    one_hot_label = paddle.nn.functional.one_hot(label, num_classes=4)
    # one_hot_label.shape = [4, 4]
    # one_hot_label = [[0., 1., 0., 0.],
    #                  [0., 1., 0., 0.],
    #                  [0., 0., 0., 1.],
    #                  [1., 0., 0., 0.]]