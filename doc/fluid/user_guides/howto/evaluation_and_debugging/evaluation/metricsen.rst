############
Model evaluation
############

Model evaluation is to use indicators to reflect the accuracy of the model under the expected target, and to determine the observation index according to the model task, as an important basis for adjusting the super-parameters in training and evaluating the effect of the model. The input to the metric function is the predicted preds and labels for the current model, and the output is custom. The metric function is very similar to the loss function, but metric is not part of the model training network.

Users can get the current predicted preds and labels through the training network, customize the metric function on the Python side, or accelerate the metric calculation on the GPU by customizing the c++ Operator.

The paddle.fluid.metrics module contains this feature


Common indicators
############

The metric function varies depending on the model task, and the indicator construction method varies depending on the task.

The regression type task labels are real numbers, so the loss and metric functions are constructed identically. Refer to the MSE method.
The commonly used indicators for classification tasks are classification indicators. The two categories mentioned in this paper are generally two-category indicators. Multi-category and multi-label need to view the corresponding API documents. For example, the ranking indicator auc, multi-classification can be used as a 0,1 classification task, and the auc indicator still applies.
Fluid contains common classification indicators, such as Precision, Recall, Accuracy, etc. Please read the API documentation for more. Take :ref:`Precision` as an example, the specific method is

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
		

For other tasks such as MultiTask Learning, Metric Learning, and Learning To Rank, please refer to the API documentation for various indicator construction methods.

Custom indicator
############
Fluid supports custom metrics and is flexible enough to support a wide range of computing tasks. The evaluation of the model is implemented below by a simple counter metric function.Where preds is the model predictor and labels is the given label.

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