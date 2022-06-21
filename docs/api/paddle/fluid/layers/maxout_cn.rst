.. _cn_api_fluid_layers_maxout:

maxout
-------------------------------

.. py:function:: paddle.fluid.layers.maxout(x, groups, name=None, axis=1)




假设输入形状为(N, Ci, H, W)，输出形状为(N, Co, H, W)，则 :math:`Co=Ci/groups` 运算公式如下：

.. math::

  y_{si+j} &= \max_k x_{gsi + sk + j} \\
  g &= groups \\
  s &= \frac{input.size}{num\_channels} \\
  0 \le &i < \frac{num\_channels}{groups} \\
  0 \le &j < s \\
  0 \le &k < groups


请参阅论文：
  - Maxout Networks:  http://www.jmlr.org/proceedings/papers/v28/goodfellow13.pdf
  - Multi-digit Number Recognition from Street View Imagery using Deep Convolutional Neural Networks: https://arxiv.org/pdf/1312.6082v4.pdf

参数
::::::::::::

    - **x** (Variable) - 4-D Tensor，maxout算子的输入张量，其数据类型为float32，数据格式为NCHW或NHWC，其中N为 batch size ，C为通道数，H和W为特征图的高和宽。
    - **groups** (int) - 指定将输入张量的channel通道维度进行分组的数目。输出的通道数量为通道数除以组数。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。 
    - **axis** (int) - 指定通道所在维度的索引。当数据格式为NCHW时，axis应该被设置为1，当数据格式为NHWC时，axis应该被设置为-1或者3。默认值：1。

返回
::::::::::::
4-D Tensor，数据类型和格式与 `x` 一致。

返回类型
::::::::::::
Variable

抛出异常
::::::::::::

    - ``ValueError`` - 如果 ``axis`` 既不是1，也不是-1或3。

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.maxout