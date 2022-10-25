.. _cn_api_fluid_layers_sigmoid_focal_loss:

sigmoid_focal_loss
-------------------------------

.. py:function:: paddle.fluid.layers.sigmoid_focal_loss(x, label, fg_num, gamma=2.0, alpha=0.25)




`Focal Loss <https://arxiv.org/abs/1708.02002>`_ 被提出用于解决计算机视觉任务中前景-背景不平衡的问题。该OP先计算输入x中每个元素的sigmoid值，然后计算sigmoid值与类别目标值label之间的Focal Loss。

Focal Loss的计算过程如下：

.. math::

  \mathop{loss_{i,\,j}}\limits_{i\in\mathbb{[0,\,N-1]},\,j\in\mathbb{[0,\,C-1]}}=\left\{
  \begin{array}{rcl}
  - \frac{1}{fg\_num} * \alpha * {(1 - \sigma(x_{i,\,j}))}^{\gamma} * \log(\sigma(x_{i,\,j})) & & {(j +1) = label_{i,\,0}}\\
  - \frac{1}{fg\_num} * (1 - \alpha) * {\sigma(x_{i,\,j})}^{ \gamma} * \log(1 - \sigma(x_{i,\,j})) & & {(j +1)!= label_{i,\,0}}
  \end{array} \right.

其中，已知：

.. math::

  \sigma(x_{i,\,j}) = \frac{1}{1 + \exp(-x_{i,\,j})}


参数
::::::::::::

    - **x**  (Variable) – 维度为 :math:`[N, C]` 的2-D Tensor，表示全部样本的分类预测值。其中，第一维N是批量内参与训练的样本数量，例如在目标检测中，样本为框级别，N为批量内所有图像的正负样本的数量总和；在图像分类中，样本为图像级别，N为批量内的图像数量总和。第二维 :math:`C` 是类别数量（ **不包括背景类** ）。数据类型为float32或float64。
    - **label**  (Variable) – 维度为 :math:`[N, 1]` 的2-D Tensor，表示全部样本的分类目标值。其中，第一维N是批量内参与训练的样本数量，第二维1表示每个样本只有一个类别目标值。正样本的目标类别值的取值范围是 :math:`[1, C]`，负样本的目标类别值是0。数据类型为int32。
    - **fg_num**  (Variable) – 维度为 :math:`[1]` 的1-D Tensor，表示批量内正样本的数量，需在进入此OP前获取正样本的数量。数据类型为int32。
    - **gamma**  (int|float) –  用于平衡易分样本和难分样本的超参数，默认值设置为2.0。
    - **alpha**  (int|float) – 用于平衡正样本和负样本的超参数，默认值设置为0.25。


返回
::::::::::::
  输入x中每个元素的Focal loss，即维度为 :math:`[N, C]` 的2-D Tensor。

返回类型
::::::::::::
 变量（Variable），数据类型为float32或float64。

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.sigmoid_focal_loss