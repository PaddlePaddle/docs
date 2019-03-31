..  _user_guide_configure_simple_model_en:

#######################
Set up Simple Model
#######################

When solving practical problems, in the beginning you can model the problem logically, and get a clear picture of **input data type** , **computing logic** , **target solution** and **optimization algorithm** of model.
PaddlePaddle provides abundant operators to implement logics of a model. In this article, we take a simple regression task as an example to clarify how to build model with PaddlePaddle.
About complete code of the example,please refer to `fit_a_line <https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/tests/book/test_fit_a_line.py>`_ 。

Description and Definition of Problem
######################################

Description : Given a pair of data :math:`<X, Y>`, figure out a function :math:`f` to make :math:`y=f(x)` . :math:`x\subset X` represents the feature of a sample, which is a real number vector with 13 dimensions; :math:`y \subset Y` is a real number representing corresponding value of the given sample.

We can try to model the problem with a regression model. Though lots of loss functions are available for regression problem, here we choose commonly used mean-square error. To simplify the problem, assuming :math:`f` is a simple linear transformation funtion, we choose random gradient descent algorithm to solve problem.

+--------------------------+-------------------------------------------------------------------------------------+
| input data type          |  sample feature: 13-dimension real number                                           |
+                          +-------------------------------------------------------------------------------------+
|                          |  sample label: 1-dimension real number                                              |
+--------------------------+-------------------------------------------------------------------------------------+
| computing logic          | use linear model to generate 1-dimensional real number as predicted output of model |
+--------------------------+-------------------------------------------------------------------------------------+
| target solution          | minimize mean-squre error between predicted output of model and sample label        |
+--------------------------+-------------------------------------------------------------------------------------+
| optimization algorithm   | random gradient descent                                                             |
+--------------------------+-------------------------------------------------------------------------------------+

Model with PaddlePaddle
#######################

After getting clear of the of input data format, model structure, loss function and optimization algorithm in terms of logic, you need to use PaddlePaddle API and operators to implement logic of model. A typical model includes four parts: format of input data, forward computing logic, loss function and optimization algorithm.

Data Layer
-----------

PaddlePaddle provides :code:`fluid.layers.data()` to describe format of input data.

The output of :code:`fluid.layers.data()` is a Variable which is in fact a Tensor. Tensor can represent multi-demensional data with its great expressive feature.In order to accurately describe data structure, it is usually necessary to indicate the shape and type of data. The shape is int vector and type can be a string. About current supported data type, please refer to    :ref:`user_guide_paddle_support_data_types_en` . Data is often read in form of batch to train model. Since batch size may vary and data operator infers batch size according to actual data, here the batch size is ignored when shape is provided. It's enough to care for the shape of single sample. For more advanced usage, please refer to :ref:`user_guide_customize_batch_size_rank_en` .  :math:`x` is real number vector of :math:`13` dimenstions while :math:`y` is a real number. Data layer can be defined as follows:

.. code-block:: python

    x = fluid.layers.data(name='x', shape=[13], dtype='float32')
    y = fluid.layers.data(name='y', shape=[1], dtype='float32')

Data in this example model are relatively simple. In fact, data operator can describe variable-length and nested sequence data. You can also use :code:`open_files` to open file to train. For more detailed documentation, please refer to :ref:`user_guide_prepare_data_en` .

Logic of Forward Computing
---------------------------

The most important part of a model is to implement logic of computing. PaddlePaddle provides lots of operators encapsulated in different granularity. These operators usually are correspondent to a kind or a group of transformation logic. The output of operator is the result of transfomation for input data. User can flexiblely use operators to implement models with complex logics. For example, many convolutional operators will be used in tasks associated with image tasks and LSTM/GRU operators will be used in sequence tasks. Various operators are usually combined in complex models to implement complex transformation. PaddlePaddle provides natural methods to combine operators. The following example displays the typical combination method:

.. code-block:: python

    op_1_out = fluid.layers.op_1(input=op_1_in, ...)
    op_2_out = fluid.layers.op_2(input=op_1_out, ...)
    ...

In the example above, op_1 and op_2 represent types of operators,such as fc performing linear transformation(full connection) or conv performing convolutional transformation. The computing order of operators and direction of data stream are defined by the connection of input and output of operators. In the example above, the output of op_1 is the input of op_2. It will firstly compute op_1 and then op_2 in the process of computing. For more complex models, we may need to use control flow operators to make it perform dynamically according to the input data. In this situation, IfElseOp, WhileOp and other operators are provided in PaddlePaddle. About documentation of these operators, please refer to :code:`fluid.layers` . As for this task, we use a fc operator:

.. code-block:: python

    y_predict = fluid.layers.fc(input=x, size=1, act=None)

Loss Function
--------------

Loss function is correspondent with the target solution. We can resolve the model by minimizing the loss value. The outputs of loss functions of most models are real numbers. But the loss operator in PaddlePaddle is only aimed at a single sample. When a batch is feeded, there will be many outputs from the loss operator, each of which is correspondent with the loss of a single sample. Therefore we usually append operators like ``mean`` after loss function to conduct reduction of losses. After each forward iteration, a loss value will be returned. After that, Chain derivation theorem will be performed automatically in PaddlePaddle to compute gradient value of every parameter and variable in computing model. Here we use mean square error cost: 

.. code-block:: python

    cost = fluid.layers.square_error_cost(input=y_predict, label=y)
    avg_cost = fluid.layers.mean(cost)

Optimization Method
---------------------

After the definition of loss function, we can get loss value by forward computing and then get gradient value of parameters with chain deravation theorem. Having obtained the gradients, parameters have to be updated and the simplest algorithm is the random gradient descent algorithm: :math:`w=w - \eta \cdot g` .But common random gradient descent algorithms have some disadvantages, such as unstable convergency. To improve the training speed and effect of model, academic scholars have come up with many optimized algorithm, including :code:`Momentum` , :code:`RMSProp` , :code:`Adam` . Strategies vary from optimization algorithm to another to update parameters of model. Usually we can choose appropriate algorthm according to specific tasks and models. No matter what optimization algorithm we adopt, learning rate is usually an important super parameter to be specified and carefully adjusted by trials. Take random gradient descent algorithm as an example here:

.. code-block:: python

    sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.001)

For more optimization operators,please refer to :code:`fluid.optimizer()` .

What to do next?
#################

Attention needs to be paid for **Data Layer**, **Forward Computing Logic**, **Loss function** and **Optimization Function** while you use PaddlePaddle to implement models.
The data format, computing logic, loss function and optimization function are all different in different tasks. A rich number of examples of model are provided in PaddlePaddle. You can build your own model structure by referring to these examples. You can visit `Model Repository <../../../user_guides/models/index_en.html>`_ to refer to examples in official documentation.
