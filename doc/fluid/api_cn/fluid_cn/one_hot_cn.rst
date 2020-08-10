.. _cn_api_fluid_one_hot:

one_hot
-------------------------------

.. py:function:: paddle.fluid.one_hot(input, num_classes, name=None)

:alias_main: paddle.nn.functional.one_hot
:alias: paddle.nn.functional.one_hot,paddle.nn.functional.common.one_hot
:old_api: paddle.fluid.one_hot


**注意：此OP要求输入Tensor shape的最后一维必须为1。
该OP将输入（input）中的每个id转换为一个one-hot向量，其长度为 ``num_classes`` ，该id对应的向量维度上的值为1，其余维度的值为0。

输出的Tensor的shape是在输入shape的最后一维后面添加了num_classes的维度。

- 示例1：

.. code-block:: python

  输入：
    X.shape = [4, 1]
    X.data = [1, 1, 3, 0]
    num_classes = 4

  输出：
    Out.shape = [4, 4]
    Out.data = [[0., 1., 0., 0.],
                [0., 1., 0., 0.],
                [0., 0., 0., 1.],
                [1., 0., 0., 0.]]


- 示例2：

.. code-block:: python
  
  输入：
    X.shape = [4, 1]
    X.data = [1, 1, 5, 0]
    num_classes = 4

  输出：抛出 Illegal value 的异常
    X中第2维的值是5，超过了num_classes，因此抛异常。  


参数：
    - **input** (Tensor) - 维度为 :math:`[N_1, ..., N_n]` 的多维Tensor，维度至少1维。数据类型为int32或int64。
    - **num_classes** (int) - 用于定义一个one-hot向量的长度。若输入为词id，则 ``num_classes`` 通常取值为词典大小。
    - **name** (str，可选) – 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回：转换后的one_hot Tensor，数据类型为float32。

返回类型：Tensor

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    # 该代码对应上述第一个示例，其中输入label的shape是[4]，输出one_hot_label的shape是[4, 4]
    label = fluid.layers.data(name="label", shape=[4], append_batch_size=False, dtype="int64")
    one_hot_label = fluid.one_hot(input=label, num_classes=4)
