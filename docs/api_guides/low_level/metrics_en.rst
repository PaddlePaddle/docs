..  _api_guide_metrics_en:


Metrics
#########
During or after the training of the neural network, it is necessary to evaluate the training effect of the model. The method of evaluation generally is calculating the distance between the overall predicted value and the overall label. Different types of tasks are applied with different evaluation methods, or with a combination of evaluation methods. In a specific task, one or more evaluation methods can be selected. Now let's take a look at commonly used evaluation methods grouped by the type of task.

Classification task evaluation
-------------------------------
The most common classification task is the binary classification task, and the multi-classification task can also be transformed into a combination of multiple binary classification tasks. The metrics commonly adopted in the two-category tasks are accuracy, correctness, recall rate, AUC and average accuracy.

- :code:`Precision` , which is used to measure the proportion of recalled ground-truth values in recalled values ​​in binary classification.

  For API Reference, please refer to :ref:`api_fluid_metrics_Precision`

- :code:`Accuracy`, which is used to measure the proportion of the recalled ground-truth value in the total number of samples in binary classification. It should be noted that the definitions of precision and accuracy are different and can be analogized to :code:`Variance` and :code:`Bias` in error analysis.

  For API Reference, please refer to :ref:`api_fluid_metrics_Accuracy`


- :code:`Recall`, which is used to measure the ratio of the recalled values to the total number of samples in binary classification. The choice of accuracy and recall rate is mutually constrained, and trade-offs are needed in the actual model. Refer to the documentation `Precision_and_recall <https://en.wikipedia.org/wiki/Precision_and_recall>`_ .

  For API Reference,  please refer to :ref:`api_fluid_metrics_Recall`

- :code:`Area Under Curve`, a classification model for binary classification, used to calculate the cumulative area of ​​the `ROC curve <https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve>`_ . :code:`Auc` is implemented via python. If you are concerned about performance, you can use :code:`fluid.layers.auc` instead.

  For API Reference,  please refer to :ref:`api_fluid_metrics_Auc`

- :code:`Average Precision`, commonly used in object detection tasks such as Faster R-CNN and SSD. The average precision is calculated under different recall conditions. For details, please refer to the document `Average precision <https://sanchom.wordpress.com/tag/average-precision/>`_ and `SSD Single Shot MultiBox Detector <https://arxiv.org/abs/1512.02325>`_ .

  For API Reference,  please refer to :ref:`api_fluid_metrics_DetectionMAP`



Sequence labeling task evaluation
----------------------------------
In the sequence labeling task, the group of tokens is called a chunk, and the model will group and classify the input tokens at the same time. The commonly used evaluation method is the chunk evaluation method.

- The chunk evaluation method :code:`ChunkEvaluator` receives the output of the :code:`chunk_eval` interface, and accumulates the statistics of chunks in each mini-batch , and finally calculates the accuracy, recall and F1 values. :code:`ChunkEvaluator` supports four labeling modes: IOB, IOE, IOBES and IO. You can refer to the documentation `Chunking with Support Vector Machines <https://aclanthology.info/pdf/N/N01/N01-1025.pdf>`_.

  For API Reference,  please refer to :ref:`api_fluid_metrics_ChunkEvaluator`


Generation/Synthesis task evaluation
----------------------------
The generation task produces output directly from the input. In NLP tasks (such as speech recognition), a new string is generated. There are several ways to evaluate the distance between a generated string and a target string, such as a multi-classification evaluation method, and another commonly used method is called editing distance.

- Edit distance: :code:`EditDistance` to measure the similarity of two strings. You can refer to the documentation `Edit_distance <https://en.wikipedia.org/wiki/Edit_distance>`_.

  For API Reference,  please refer to :ref:`api_fluid_metrics_EditDistance`
