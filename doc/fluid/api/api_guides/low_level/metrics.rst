..  _api_guide_optimizer:


Metrics
#########

在神经网络训练过程或者训练完成后，需要评估模型的训练效果，评估的方法一般是计算预测值和真值(label)之间的距离，不同模型会用不同的度量方法，比如分类模型常用AUC作为分类效果的度量.

1.MetricBase
------------------

:code:`MetricBase` 是所有度量类的基类，定义了实现度量计算的一组接口方法。由于度量值是全局指标，所以需要在每一个minibatch训练完成后，使用 :code:`update`接口更新全局指标状态，这是个累积过程，而累积是从调用 :code:`reset`方法开始的，最后使用:code:`eval`接口计算累计的度量值。如果希望实现自定义的度量计算，需要继承:code:`MetricBase`类，然后实现接口方法。

API Reference 请参考 :ref:`api_fluid_metcics_MetricBase`

2.CompositeMetric
------------------

:code:`CompositeMetric` 可以组合多个度量指标，只需要在每一个step提供一次预测值和真值，就可以获得多个指标值。

API Reference 请参考 :ref:`api_fluid_metcics_CompositeMetric`

3.Precision
------------------

:code:`Precision`是准确率，用来衡量二分类中召回真值和召回值的比例。需要注意的是，准确率和正确率的定义是不同的，正确率是二分类中召回真值和总样本数的比例。

API Reference 请参考 :ref:`api_fluid_metcics_Precision`

4.Recall
------------------

:code:`Recall`是召回率，用来衡量二分类中召回值和总样本数的比例。可以参考 https://en.wikipedia.org/wiki/Precision_and_recall

API Reference 请参考 :ref:`api_fluid_metcics_Recall`

5.Accuracy
------------------

:code:`Accuracy`是正确率，用来衡量二分类中二分类中召回真值和总样本数的比例。

API Reference 请参考 :ref:`api_fluid_metcics_Accuracy`

6.ChunkEvaluator
------------------

:code:`ChunkEvaluator`是分组评估方法，通过接收chunk_eval方法的输出，累积每一个step的分组数据计算准确率、召回率和F1值。:code:`ChunkEvaluator`支持IOB, IOE, IOBES and IO四种标注模式。基本原来可以参考https://aclanthology.info/pdf/N/N01/N01-1025.pdf

API Reference 请参考 :ref:`api_fluid_metcics_Accuracy`


7.EditDistance
------------------

:code:`EditDistance`字符串编辑距离，用来衡量两个字符串的相似度。可以参考https://en.wikipedia.org/wiki/Edit_distance

API Reference 请参考 :ref:`api_fluid_metcics_EditDistance`


8.DetectionMAP
------------------

:code:`DetectionMAP `是检测平均准确度，用在Faster R-CNN和SSD等物体检测模型中度量准确率。在不同召回条件下，计算了最大准确率的平均值，具体可以参考
https://sanchom.wordpress.com/tag/average-precision/  
https://arxiv.org/abs/1512.02325

API Reference 请参考 :ref:`api_fluid_metcics_DetectionMAP`



9.Auc
------------------

:code:`Auc`适用于二分类的分类模型评估，用来计算ROC曲线的累积面积。这个度量方法适用python实现，如果你比较关注性能，可以使用fluid.layers.auc代替。原理可以参考https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve 

API Reference 请参考 :ref:`api_fluid_metcics_Auc`


