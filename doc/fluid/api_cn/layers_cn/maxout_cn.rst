.. _cn_api_fluid_layers_maxout:

maxout
-------------------------------

.. py:function:: paddle.fluid.layers.maxout(x, groups, name=None)

假设输入形状为(N, Ci, H, W)，输出形状为(N, Co, H, W)，则 :math:`Co=Ci/groups` 运算公式如下:

.. math::

  y_{si+j} &= \max_k x_{gsi + sk + j} \\
  g &= groups \\
  s &= \frac{input.size}{num\_channels} \\
  0 \le &i < \frac{num\_channels}{groups} \\
  0 \le &j < s \\
  0 \le &k < groups


请参阅论文:
  - Maxout Networks:  http://www.jmlr.org/proceedings/papers/v28/goodfellow13.pdf
  - Multi-digit Number Recognition from Street View Imagery using Deep Convolutional Neural Networks: https://arxiv.org/pdf/1312.6082v4.pdf

参数：
    - **x** (Variable) - 维度为：math:`[N,C,H,W]`的4-D Tensor，其中N为 batch size ，C为通道数，H和W为特征图的高和宽。数据类型为float32。maxout算子的输入张量。
    - **groups** （int32）- 指定将输入张量的channel通道维度进行分组的数目。输出的通道数量为通道数除以组数。
    - **name** (str|None) – 该层的名称（可选项）,默认为None


返回：表示为输出的Tensor，数据类型为float32。输出维度也是NCHW。其中N为 batch size，C为通道数，H和W为特征的高和宽。


返回类型：Variable


**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    input = fluid.layers.data(
        name='data',
        shape=[256, 32, 32],
        dtype='float32')
    out = fluid.layers.maxout(input, groups=2)
