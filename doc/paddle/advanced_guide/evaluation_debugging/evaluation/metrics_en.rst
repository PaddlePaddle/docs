################
Model Evaluation
################

Model evaluation is to use metrics to reflect the accuracy of the model under the expected target. The metrics are determined by model tasks. Model evaluation is an important basis for adjusting the super-parameters in training and evaluating the effect of the model. The input to the metric function is the predicted preds and labels for the current model, and the output is customized. The metric function is very similar to the loss function, but metric is not a component of the model training network.

Users can get the current predicted preds and labels through training network, and customize the metric function on the Python side, or accelerate the metric calculation on the GPU by customizing the C++ Operator.

The ``paddle.fluid.metrics`` module contains this feature.


Common metrics
##################

The metric function varies with different model tasks, and so does the metric construction.

The labels in regression task are real numbers, you can refer to the MSE (Mean Squared Error) method for help.
The commonly used metrics for classification tasks are classification metrics. The metric function mentioned in this paper is generally metrics of binary classification. For details of metrics for multi-category and multi-label tasks, please read the corresponding API documents. For example, the ranking metric auc function works for multi-classification tasks because these tasks can be used as a 0,1 classification task.
Fluid contains common classification metrics, such as Precision, Recall, Accuracy, etc. Please read the API documentation for more. Take ``Precision`` as an example, the specific method is

.. code-block:: python


   import paddle.fluid as fluid
   import numpy as np

   metric = fluid.metrics.Precision()

   # generate the preds and labels

   preds = [[0.1], [0.7], [0.8], [0.9], [0.2],
            [0.2], [0.3], [0.5], [0.8], [0.6]]

   labels = [[0], [1], [1], [1], [1],
             [0], [0], [0], [0], [0]]

   preds = np.array(preds)
   labels = np.array(labels)

   metric.update(preds=preds, labels=labels)
   numpy_precision = metric.eval()

   print("expect precision: %.2f and got %.2f" % (3.0 / 5.0, numpy_precision))


As for other tasks such as MultiTask Learning, Metric Learning, and Learning To Rank, please refer to the API documentation for their various metric construction methods.

Custom metrics
################
Fluid supports custom metrics and is flexible enough to support a wide range of computing tasks. The evaluation of the model is implemented below with a metric function composed of a simple counter, where ``preds`` is the prediction values and ``labels`` is the given labels.

.. code-block:: python

   class MyMetric(MetricBase):
       def __init__(self, name=None):
           super(MyMetric, self).__init__(name)
           self.counter = 0  # simple counter

       def reset(self):
           self.counter = 0

       def update(self, preds, labels):
           if not _is_numpy_(preds):
               raise ValueError("The 'preds' must be a numpy ndarray.")
           if not _is_numpy_(labels):
               raise ValueError("The 'labels' must be a numpy ndarray.")
           self.counter += sum(preds == labels)

       def eval(self):
           return self.counter
