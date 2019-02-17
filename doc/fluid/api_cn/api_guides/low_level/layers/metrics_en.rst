..  _api_guide_metrics_en:


Metrics
#########
During the training of the neural network or after the finish of training, it is necessary to evaluate the training effect of the model. The method of evaluation generally is calculating the distance between the overall predicted value and the overall label. Different types of tasks are applied with different evaluation methods, or with a plurality of evaluation methods in combination. In a specific task, one or more evaluation methods can be selected. The following commonly used evaluation methods are introduced according to the type of task.

Classification Task Evaluation
-------------------------------
The most commonly used classification task is the bipartition task, and the multi-category task can also be transformed into a combination of multiple two-category tasks. The evaluation indexes commonly used in the two-category tasks are accuracy, correctness, recall rate, AUC and average accuracy.

- Precision: :code:`Precision` ,which is used to measure the proportion of recalled true values and recall values ​​in the second category.

  About API Reference, please refer to :ref:`api_fluid_metrics_Precision`

- Accuracy: :code:`Accuracy`, which is used to measure the ratio of the recalled true value to the total number of samples in the second category. It should be noted that the definitions of precision and accuracy are different and can be analogized to :code:`Variance` and :code:`Bias` in error analysis.

  About API Reference, please refer to :ref:`api_fluid_metrics_Accuracy`


- Recall: :code:`Recall`, which is used to measure the ratio of the recall value to the total number of samples in the bipartition task. The choice of accuracy and recall rate is mutually constrained, and trade-offs are needed in the actual model. Refer to the documentation `Precision_and_recall <https://en.wikipedia.org/wiki/Precision_and_recall>`_ .

  About API Reference,  please refer to :ref:`api_fluid_metrics_Recall`

- AUC: :code:`Area Under Curve`, a classification model for bipartition, used to calculate the cumulative area of ​​the `ROC curve <https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve>`_. :code:`Auc` is implemented via python. If you are concerned about performance, you can use :code:`fluid.layers.auc` instead.

  About API Reference,  please refer to :ref:`api_fluid_metrics_Auc`

- Average accuracy: :code:`Average Precision`, commonly used in object detection tasks such as Faster R-CNN and SSD. The average of the precision is calculated under different recall conditions. For details, please refer to the document `Average-precision <https://sanchom.wordpress.com/tag/average-precision/>`_ and `SSD: Single Shot MultiBox Detector <https://arxiv.org/abs/1512.02325>`_.

  About API Reference,  please refer to :ref:`api_fluid_metrics_DetectionMAP`



Sequence labeling task evaluation
----------------------------------
In the sequence labeling task, the group of tokens is called a chunk, and the model will group and classify the input tokens at the same time. The commonly used evaluation method is the chunk evaluation method.

- The block evaluation method: :code:`ChunkEvaluator` , receives the output of the :code:`chunk_eval` interface, accumulates the statistics of each minibatch block, and finally calculates the accuracy, recall and F1 values. :code:`ChunkEvaluator` supports four annotation modes: IOB, IOE, IOBES and IO. You can refer to the documentation `Chunking with Support Vector Machines <https://aclanthology.info/pdf/N/N01/N01-1025.pdf>`_.

  About API Reference,  please refer to :ref:`api_fluid_metrics_ChunkEvaluator`


Generate task evaluation
----------------------------
The generated task produces output directly from the input. Corresponding to NLP tasks (such as speech recognition), a new string is generated. There are several ways to evaluate the distance between a generated string and a target string, such as a multi-classification evaluation method, and another commonly used method is called editing distance.

- Edit distance: :code:`EditDistance` to measure the similarity of two strings. You can refer to the documentation `Edit_distance <https://en.wikipedia.org/wiki/Edit_distance>`_.

  About API Reference,  please refer to :ref:`api_fluid_metrics_EditDistance`