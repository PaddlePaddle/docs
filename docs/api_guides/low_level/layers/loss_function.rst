..  _api_guide_loss_function:

############
损失函数
############

损失函数定义了拟合结果和真实结果之间的差异，作为优化的目标直接关系模型训练的好坏，很多研究工作的内容也集中在损失函数的设计优化上。
Paddle 中提供了面向多种任务的多种类型的损失函数，以下列出了一些 Paddle 中包含的较为常用的损失函数。

回归
======

平方误差损失（squared error loss）使用预测值和真实值之间误差的平方作为样本损失，是回归问题中最为基本的损失函数。
API Reference 请参考 :ref:`cn_api_paddle_nn_functional_square_error_cost`。

平滑 L1 损失（smooth_l1 loss）是一种分段的损失函数，较平方误差损失其对异常点相对不敏感，因而更为鲁棒。
API Reference 请参考 :ref:`cn_api_paddle_nn_functional_smooth_l1_loss`。


分类
======

`交叉熵（cross entropy） <https://en.wikipedia.org/wiki/Cross_entropy>`_ 是分类问题中使用最为广泛的损失函数，Paddle 中提供了接受归一化概率值和非归一化分值输入的两种交叉熵损失函数的接口，并支持 soft label 和 hard label 两种样本类别标签。
API Reference 请参考 :ref:`cn_api_paddle_nn_functional_cross_entropy`。

多标签分类
---------------------
对于多标签分类问题，如一篇文章同属于政治、科技等多个类别的情况，需要将各类别作为独立的二分类问题计算损失，Paddle 中为此提供了 binary_cross_entropy 损失函数，
API Reference 请参考 :ref:`cn_api_paddle_nn_functional_binary_cross_entropy`。

大规模分类
---------------------
对于大规模分类问题，通常需要特殊的方法及相应的损失函数以加速训练，常用的方法有 `噪声对比估计（Noise-contrastive estimation，NCE） <http://proceedings.mlr.press/v9/gutmann10a/gutmann10a.pdf>`_ 和 `层级 sigmoid <http://www.iro.umontreal.ca/~lisa/pointeurs/hierarchical-nnlm-aistats05.pdf>`_ 。

* 噪声对比估计通过将多分类问题转化为学习分类器来判别数据来自真实分布和噪声分布的二分类问题，基于二分类来进行极大似然估计，避免在全类别空间计算归一化因子从而降低了计算复杂度。
* 层级 sigmoid 通过二叉树进行层级的二分类来实现多分类，每个样本的损失对应了编码路径上各节点二分类交叉熵的和，避免了归一化因子的计算从而降低了计算复杂度。

这两种方法对应的损失函数在 Paddle 中均有提供，API Reference 请参考 :ref:`cn_api_paddle_static_nn_nce` 和 :ref:`cn_api_paddle_nn_functional_hsigmoid_loss`。

序列分类
---------------------
序列分类可以分为以下两种：

* 序列分类（Sequence Classification）问题，整个序列对应一个预测标签，如文本分类。这种即是普通的分类问题，可以使用 cross entropy 作为损失函数。
* 时序分类（Temporal Classification）问题，需要对未分割的序列进行标注，如语音识别。对于这种时序分类问题，`CTC（Connectionist Temporal Classification） <http://people.idsia.ch/~santiago/papers/icml2006.pdf>`_ 损失函数不需要对齐输入数据及标签，可以进行端到端的训练，Paddle 提供了 warpctc 的接口来计算相应的损失，API Reference 请参考 :ref:`cn_api_paddle_nn_functional_ctc_loss`。

排序
======

`排序问题 <https://en.wikipedia.org/wiki/Learning_to_rank>`_ 可以使用 Pointwise、Pairwise 和 Listwise 的学习方法，不同的方法需要使用不同的损失函数：

* Pointwise 的方法通过近似为回归问题解决排序问题，可以使用回归问题的损失函数。
* Pairwise 的方法需要特殊设计的损失函数，其通过近似为分类问题解决排序问题，使用两篇文档与 query 的相关性得分以偏序作为二分类标签来计算损失。Paddle 中提供了一种常用的 Pairwise 方法的损失函数，API Reference 请参考 :ref:`cn_api_paddle_nn_functional_margin_ranking_loss`。

更多
======

对于一些较为复杂的损失函数，可以尝试使用其他损失函数组合实现；Paddle 中提供的用于图像分割任务的 :ref:`cn_api_paddle_nn_functional_dice_loss` 即是使用其他 OP 组合（计算各像素位置似然概率的均值）而成；多目标损失函数也可看作这样的情况，如 Faster RCNN 就使用 cross entropy 和 smooth_l1 loss 的加权和作为损失函数。

**注意**，在定义损失函数之后为能够使用 :ref:`api_guide_optimizer` 进行优化，通常需要使用 :ref:`cn_api_paddle_mean` 或其他操作将损失函数返回的高维 Tensor 转换为 Scalar 值。
