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
    - **x** (Variable) - (tensor) maxout算子的输入张量。输入张量的格式为NCHW。其中N为 batch size ，C为通道数，H和W为feature的高和宽
    - **groups** （INT）- 指定将输入张量的channel通道维度进行分组的数目。输出的通道数量为通道数除以组数。
    - **name** (basestring|None) - 输出的名称

返回：Tensor，maxout算子的输出张量。输出张量的格式也是NCHW。其中N为 batch size，C为通道数，H和W为特征的高和宽。

返回类型：out（Variable）

**代码示例**：
    
.. code-block:: python

    import paddle.fluid as fluid
    input = fluid.layers.data(
        name='data',
        shape=[256, 32, 32],
        dtype='float32')
    out = fluid.layers.maxout(input, groups=2)










