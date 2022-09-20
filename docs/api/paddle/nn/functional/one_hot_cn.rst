.. _cn_api_nn_functional_one_hot:

one_hot
-------------------------------

.. py:function:: paddle.nn.functional.one_hot(x, num_classes, name=None)
将输入'x'中的每个 id 转换为一个 one-hot 向量，其长度为 ``num_classes``，该 id 对应的向量维度上的值为 1，其余维度的值为 0。

输出的 Tensor 的 shape 是在输入 shape 的最后一维后面添加了 num_classes 的维度。

- 示例 1：

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

- 示例 2：

.. code-block:: text

  输入：
    X.shape = [4]
    X.data = [1, 1, 5, 0]
    num_classes = 4

  输出：抛出 Illegal value 的异常
    X 中第 2 维的值是 5，超过了 num_classes，因此抛异常。


参数
::::::::::::

    - **x** (Tensor) - 维度为 :math:`[N_1, ..., N_n]` 的多维 Tensor，维度至少 1 维。数据类型为 int32 或 int64。
    - **num_classes** (int) - 用于定义一个 one-hot 向量的长度。若输入为词 id，则 ``num_classes`` 通常取值为词典大小。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
Tensor，转换后的 one_hot Tensor，数据类型为 float32。

代码示例
::::::::::::

COPY-FROM: paddle.nn.functional.one_hot
