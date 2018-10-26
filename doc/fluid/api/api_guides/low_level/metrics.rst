..  _api_guide_optimizer:


Metrics
#########

在神经网络训练过程中或者训练完成后，需要评估模型的训练效果，评估的方法一般是计算全体预测值和全体真值(label)之间的距离，不同模型会用不同的度量方法，比如分类模型常用 :code:`AUC` 作为分类效果的度量, OCR模型可以用 :code:`EditDistance` 作为识别效果的度量。

1.MetricBase
------------------

:code:`MetricBase` 是所有度量类的基类，定义了实现度量计算的一组接口方法。由于度量值是全局指标，所以需要在每一个minibatch训练完成后，使用 :code:`update` 接口更新全局指标状态，这是个累积过程，而累积是从调用 :code:`reset` 方法开始的，最后使用 :code:`eval` 接口计算累计的度量值。如果希望实现自定义的度量计算，需要继承 :code:`MetricBase` 类，然后实现其接口方法。

API Reference 请参考 :ref:`api_fluid_metrics_MetricBase`

2.CompositeMetric
------------------

:code:`CompositeMetric` 可以组合多个度量指标，只需要在每一个step提供一次预测值和真值，就可以获得多个指标值。

API Reference 请参考 :ref:`api_fluid_metrics_CompositeMetric`

3.Precision/Accuracy/Recall/Auc
------------------

:code:`Precision` 是准确率，用来衡量二分类中召回真值和召回值的比例。:code:`Accuracy` 是正确率，用来衡量二分类中二分类中召回真值和总样本数的比例。需要注意的是，准确率和正确率的定义是不同的，区别可以类比于误差分析中的 :code:`Variance` 和 :code:`Bias` 。:code:`Recall` 是召回率，用来衡量二分类中召回值和总样本数的比例。准确率和召回率的选取相互制约，实际模型中需要进行权衡，可以参考文档 `Precision_and_recall <https://en.wikipedia.org/wiki/Precision_and_recall>`_ 。
:code:`Auc` 适用于二分类的分类模型评估，用来计算 `ROC曲线的累积面积 <https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve>`_。:code:`Auc` 通过python计算实现，如果关注性能，可以使用 :code:`fluid.layers.auc` 代替。

API Reference 请参考 

:ref:`api_fluid_metrics_Precision` 

:ref:`api_fluid_metrics_Accuracy` 

:ref:`api_fluid_metrics_Recall` 

:ref:`api_fluid_metrics_Auc`

4.ChunkEvaluator
------------------

:code:`ChunkEvaluator` 是分组评估度量，接收 :code:`chunk_eval` 接口的输出，累积每一个minibatch的分组统计，最后计算准确率、召回率和F1值。:code:`ChunkEvaluator` 支持IOB, IOE, IOBES and IO四种标注模式。可以参考文档 `Chunking with Support Vector Machines <https://aclanthology.info/pdf/N/N01/N01-1025.pdf>`_ 

API Reference 请参考 :ref:`api_fluid_metrics_ChunkEvaluator`


5.EditDistance
------------------

:code:`EditDistance` 字符串编辑距离，用来衡量两个字符串的相似度。可以参考文档 `Edit_distance <https://en.wikipedia.org/wiki/Edit_distance>`_

API Reference 请参考 :ref:`api_fluid_metrics_EditDistance`


6.DetectionMAP
------------------

:code:`DetectionMAP` 是检测平均准确度，用在Faster R-CNN和SSD等物体检测模型中度量准确率。在不同召回条件下，计算了最大准确率的平均值，具体可以参考文档
`Average-precision <https://sanchom.wordpress.com/tag/average-precision/>`_ 和 `SSD: Single Shot MultiBox Detector <https://arxiv.org/abs/1512.02325>`_

API Reference 请参考 :ref:`api_fluid_metrics_DetectionMAP`







