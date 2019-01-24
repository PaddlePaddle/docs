################
Model Evaluation
################

Model evaluation is to use indicators to reflect the accuracy of the model under the expected target, and to determine the observation index according to the model task, as an important basis for adjusting the super-parameters in training and evaluating the effect of the model. The input to the metric function is the predicted preds and labels for the current model, and the output is customized. The metric function is very similar to the loss function, but metric is not a component of the model training network.

Users can get the current predicted preds and labels through training network, and customize the metric function on the Python side, or accelerate the metric calculation on the GPU by customizing the C++ Operator.

The ``paddle.fluid.metrics`` module contains this feature.


Common metrics
##################

The metric function varies with different model tasks, and so does the metric construction.

The labels in regression task are real numbers, so the loss and metric functions are constructed in the same way， you can refer to the MSE method for help.
The commonly used metrics for classification tasks are classification indicators.The indicator mentioned in this paper is generally metrics of binary classification. For details of indicators of multi-category and multi-label tasks, please read the corresponding API documents. For example, the ranking metric auc, multi-classification can be used as a 0,1 classification task, and the auc indicator still works.
Fluid contains common classification metrics, such as Precision, Recall, Accuracy, etc. Please read the API documentation for more. Take :ref:`Precision` as an example, the specific method is

.. code-block:: python

	>>> import paddle.fluid as fluid
	>>> labels = fluid.layers.data(name="data", shape=[1], dtype="int32")
	>>> data = fluid.layers.data(name="data", shape=[32, 32], dtype="int32")
	>>> pred = fluid.layers.fc(input=data, size=1000, act="tanh")
	>>> acc = fluid.metrics.Precision()
	>>> for pass in range(PASSES):
	>>> acc.reset()
	>>> for data in train_reader():
	>>> loss, preds, labels = exe.run(fetch_list=[cost, preds, labels])
	>>> acc.update(preds=preds, labels=labels)
	>>> numpy_acc = acc.eval()
		

For other tasks such as MultiTask Learning, Metric Learning, and Learning To Rank, please refer to the API documentation for various metric construction methods.

Custom metrics
################
Fluid supports custom metrics and is flexible enough to support a wide range of computing tasks. The evaluation of the model is implemented below with a metric function composed of a simple counter metric function, where ``preds`` is the prediction values and ``labels`` is the given labels.

.. code-block:: python

	>>> class MyMetric(MetricBase):
	>>> def __init__(self, name=None):
	>>> super(MyMetric, self).__init__(name)
	>>> self.counter = 0 # simple counter
	>>> def reset(self):
	>>> self.counter = 0

	>>> def update(self, preds, labels):
	>>> if not _is_numpy_(preds):
	>>> raise ValueError("The 'preds' must be a numpy ndarray.")
	>>> if not _is_numpy_(labels):
	>>> raise ValueError("The 'labels' must be a numpy ndarray.")
	>>> self.counter += sum(preds == labels)

	>>> def eval(self):
	>>> return self.counter
