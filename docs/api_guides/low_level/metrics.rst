..  _api_guide_metrics:


评价指标
#########
在神经网络训练过程中或者训练完成后，需要评价模型的训练效果。评价的方法一般是计算全体预测值和全体真值(label)之间的距离，不同类型的任务会使用不同的评价方法，或者综合使用多个评价方法。在具体的任务中，可以选用一种或者多种评价方法。下面对常用的评价方法按照任务类型做介绍。

分类任务评价
------------------
分类任务中最常用的是二分类，而多分类任务也可以转化为多个二分类任务的组合，二分类任务常用的评价指标有准确率、正确率、召回率、AUC 和平均准确度。

- 准确率: :code:`Precision` ，用来衡量二分类中召回真值和召回值的比例。

  API Reference 请参考 :ref:`cn_api_fluid_metrics_Precision`

- 正确率: :code:`Accuracy` ，用来衡量二分类中召回真值和总样本数的比例。需要注意的是，准确率和正确率的定义是不同的，可以类比于误差分析中的 :code:`Variance` 和 :code:`Bias` 。

  API Reference 请参考 :ref:`cn_api_fluid_metrics_Accuracy`


- 召回率: :code:`Recall` ，用来衡量二分类中召回值和总样本数的比例。准确率和召回率的选取相互制约，实际模型中需要进行权衡，可以参考文档 `Precision_and_recall <https://en.wikipedia.org/wiki/Precision_and_recall>`_ 。

  API Reference 请参考 :ref:`cn_api_fluid_metrics_Recall`

- AUC: :code:`Area Under Curve`， 适用于二分类的分类模型评估，用来计算 `ROC 曲线的累积面积 <https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve>`_。:code:`Auc` 通过 python 计算实现，如果关注性能，可以使用 :code:`fluid.layers.auc` 代替。

  API Reference 请参考 :ref:`cn_api_fluid_metrics_Auc`

- 平均准确度: :code:`Average Precision` ，常用在 Faster R-CNN 和 SSD 等物体检测任务中。在不同召回条件下，计算了准确率的平均值，具体可以参考文档 `Average-precision <https://sanchom.wordpress.com/tag/average-precision/>`_ 和 `SSD: Single Shot MultiBox Detector <https://arxiv.org/abs/1512.02325>`_。

  API Reference 请参考 :ref:`cn_api_fluid_metrics_DetectionMAP`



序列标注任务评价
------------------
序列标注任务中，token 的分组称为语块(chunk)，模型会同时将输入的 token 分组和分类，常用的评估方法是语块评估方法。

- 语块评估方法: :code:`ChunkEvaluator` ，接收 :code:`chunk_eval` 接口的输出，累积每一个 minibatch 的语块统计值，最后计算准确率、召回率和 F1 值。:code:`ChunkEvaluator` 支持 IOB, IOE, IOBES 和 IO 四种标注模式。可以参考文档 `Chunking with Support Vector Machines <https://aclanthology.info/pdf/N/N01/N01-1025.pdf>`_ 。

  API Reference 请参考 :ref:`cn_api_fluid_metrics_ChunkEvaluator`


生成任务评价
------------------
生成任务会依据输入直接产生输出。对应 NLP 任务中(比如语音识别)，则生成新字符串。评估生成字符串和目标字符串之间距离的方法也有多种，比如多分类评估方法，而另外一种常用的方法叫做编辑距离。

- 编辑距离: :code:`EditDistance` ，用来衡量两个字符串的相似度。可以参考文档 `Edit_distance <https://en.wikipedia.org/wiki/Edit_distance>`_。

  API Reference 请参考 :ref:`cn_api_fluid_metrics_EditDistance`
