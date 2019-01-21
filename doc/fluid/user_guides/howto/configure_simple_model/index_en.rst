..  _user_guide_configure_simple_model:

#######################
Configure Simple Model
#######################

You can model the problem logically when you solve the real problem,Make clear  **input data type**, **computing logic**, **solving problem** and **optimized algorithm** of model.
PaddlePaddle provide lots of operators to implement logic of model. Take a simple regression task as an example to clarify how to build model with PaddlePaddle.
About complete code of the example,please refer to `fit_a_line <https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/tests/book/test_fit_a_line.py>`_。

Description and Definition of Problem
######################################

Description of the problem : given a pair of data :math:`<X, Y>`,figure out a function :math:`f` to make :math:`y=f(x)` . :math:`x\subset X` represent the feature of a sample as well as a real number vector with dimension of :math:`13` ; :math:`y \subset Y` is a real number representing value of the sample.

We can try to model the problem with regression model.Though lots of loss functions are available for regression problem,here we choose commonly used mean-square error.To simplify the problem,assuming :math:`f` is a simple linear transfomed funtion,we choose random gradient descent algorithm to solve problem.

+-----------------------+-------------------------------------------------------------------------------------+
| input data type       |  sample feature: 13 dimension real number                                           |
+                       +-------------------------------------------------------------------------------------+
|                       |  sample label: 1 dimension real number                                              |
+-----------------------+-------------------------------------------------------------------------------------+
| computing logic       | use linear model to generate 1 dimensional real number as predicted output of model |
+-----------------------+-------------------------------------------------------------------------------------+
| solving problem       | minimize mean-squre error between predicted output of model and sample label        |
+-----------------------+-------------------------------------------------------------------------------------+
| optimized algorithm   | random gradient descent                                                             |
+-----------------------+-------------------------------------------------------------------------------------+

Model with PaddlePadle
#######################

After making clear format of input data,model structure,loss function and optimized algorithm in terms of logic, you need to use PaddlePaddle API and operators to implement logic of model.A typical model includes four parts:format of input data,forward computing logic,loss function and optimized algorithm.

Data Layer
------

PaddlePaddle provides :code:`fluid.layers.data()` to describe format of input data.

The ouput of :code:`fluid.layers.data()` is a Variable which is in fact a Tensor。Tensor can represent multi-demensional data with its great expressive feature.In order to accurately describe data structure, it usually needs to indicate shape(data) and type(data type).The shape is int vector and type can be a string.About current supported data type,please refer to    :ref:`user_guide_paddle_support_data_types` .Data is often read in form of batch to train model.Since batch size is variable and data operator infers batch size according to real data,you can ignore batch size while you assign shape.It's enough for shape of one sample.About more advanced usage,please refer to :ref:`user_guide_customize_batch_size_rank`.  :math:`x` is real number vector of :math:`13` dimenstion while :math:`y` is real number. Data layer can be defined as follows:

.. code-block:: python

    x = fluid.layers.data(name='x', shape=[13], dtype='float32')
    y = fluid.layers.data(name='y', shape=[1], dtype='float32')

Data in model is relatively simple. In fact,data operator can describe variable-length and nested sequence data. You can also use :code:`open_files` to open file to train. About more detailed documents,please refer to :ref:`user_guide_prepare_data` .

Logic of Forward Computing
---------------------------

The most important part of a model is to implement logic of computing. PaddlePaddle provides lots of operators packaged with different grains.There operators usually are correspondent to a kind or a group of logics of transformation. The output of operator is the result of transfomation for input data. User can flexiblely use operators to implement models with complex logics. For example,many convolutional operators will be used in tasks associated with image tasks and many LSTM/GRU operators will be used in sequence tasks. Various operators are usually combined in complex models to implement complex transformation. PaddlePaddle provides natural and influent methods to combine operators as follows:

.. code-block:: python

    op_1_out = fluid.layers.op_1(input=op_1_in, ...)
    op_2_out = fluid.layers.op_2(input=op_1_out, ...)
    ...

op_1 and op_2 represent types of operators,such as fc performing linear transformation(full connection) or conv performing convolutional transformation and so on. The computing order of operators and direction of data stream are defined by the connection of input and output of operators. In examples above,the output of op_1 is the input of op_2. It will compute op_1 and then op_2 in the process of computing. For more complex models,we may need to use control stream operator to make it performed dynamically according to input data. In this situation, IfElseOp, WhileOp and other operators are provided in PaddlePaddle. About document of operators,please refer to :code:`fluid.layers` . As for this task,we use a fc operator:

.. code-block:: python

    y_predict = fluid.layers.fc(input=x, size=1, act=None)

Loss Function
--------------

Loss function is correspondent with problem to be solved.We can figure out the model by minimizing loss. The outputs of loss function of most models are real number.But the loss operator in PaddlePaddle is only for one sample.When a batch is feeded, there are many outputs of loss operator, each of which is correspondent with the loss of a sample. Therefore we usually use operators like mean to reduce losses. Chain derivation theorem will be performed automatically in PaddlePaddle to compute gradient value of every parameter and variable in computing model. Here we use mean square error cost: 

.. code-block:: python

    cost = fluid.layers.square_error_cost(input=y_predict, label=y)
    avg_cost = fluid.layers.mean(cost)

Optimized Method
-----------------

After the definition of loss function,we can get loss value by forward computing and then get gradient value of parameters with chain deravation theorem.Then parameters have to be updated and the most simple algorithm is random random gradient descent algorithm: :math:`w=w - \eta \cdot g`.But common random gradient descent algorithms exist some disadvantages,such as unstable convergency.To improve the train speed and result of model, academic members come up with many optimized algorithm,including :code:`Momentum`、:code:`RMSProp`、:code:`Adam` and so on. Strategies vary from optimized algorithm to optimized algorithm to update parameters of model. Usually we can choose appropriate algorthm according to specific tasks and models. No matter what optimized algorithm we adopt,learning rate is usually an important super parameter to be indicated and careful adjustment by trials. Take random gradient algorithm as an example here:

.. code-block:: python

    sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.001)

About more optimized operators,please refer to :code:`fluid.optimizer()` .

What to do next?
#################

Attention needs to be paid for **Data Layer**, **Forward Computing Logic**, **Loss function** and **Optimized Function** while you use PaddlePaddle to implement models.
The data format,computing logic,loss function and optimized function are all different in different tasks. Many examples of model are provided in PaddlePaddle. You can build your own model structure in reference to these examples. You can visit `Model Repository <https://github.com/PaddlePaddle/models/tree/develop/fluid>`_ to refer to examples in offical document.
