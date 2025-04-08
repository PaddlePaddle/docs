..  _api_guide_optimizer_en:

###########
Optimizer
###########

Neural network in essence is a `Optimization problem <https://en.wikipedia.org/wiki/Optimization_problem>`_ .
With `forward computing and back propagation <https://zh.wikipedia.org/zh-hans/backpropagation_algorithm>`_ ,
:code:`Optimizer` use back-propagation gradients to optimize parameters in a neural network.


1. Adam
------------------------
Optimizer of `Adam <https://arxiv.org/abs/1412.6980>`_ is a method to adaptively adjust learning rate,
fit for most `non-convex optimization <https://www.cs.cornell.edu/courses/cs6787/2017fa/Lecture7.pdf>`_ , big data set and high-dimensional scenarios. :code:`Adam` is the most common optimization algorithm.

API Reference: :ref:`api_paddle_optimizer_Adam`


2. SGD
------------------------
:code:`SGD` is an offspring class of :code:`Optimizer` implementing `Stochastic Gradient Descent <https://arxiv.org/pdf/1609.04747>`_ which is a method of `Gradient Descent <https://en.wikipedia.org/wiki/Gradient_descent>`_ .
When it needs to train a large number of samples, we usually choose :code:`SGD` to make loss function converge more quickly.

API Reference: :ref:`api_paddle_optimizer_SGD`


3. Momentum
------------------------
:code:`Momentum` optimizer adds momentum on the basis of :code:`SGD` , reducing noise problem in the process of random gradient descent.
You can set :code:`use_nesterov` as False or True, respectively corresponding to traditional `Momentum(Section 4.1 in thesis)
<https://arxiv.org/pdf/1609.04747>`_  algorithm and `Nesterov accelerated gradient(Section 4.2 in thesis)
<https://arxiv.org/pdf/1609.04747>`_ algorithm.

API Reference: :ref:`api_paddle_optimizer_Momentum`


4. AdamW
------------------------
`AdamW <https://arxiv.org/abs/1711.05101>`_ optimizer is an improved version of :code:`Adam`, which decouples weight decay (regularization) from gradient updates, addressing the issue of L2 regularization failure in :code:`Adam`.

API Reference: :ref:`api_paddle_optimizer_AdamW`


5. Adagrad
------------------------
`Adagrad <http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf>`_ optimizer can adaptively allocate different learning rates for parameters to solve the problem of different sample sizes for different parameters.

API Reference: :ref:`api_paddle_optimizer_Adagrad`


6. RMSProp
------------------------
`RMSProp <http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf>`_ optimizer is a method to adaptively adjust learning rate.
It mainly solves the problem of dramatic decrease of learning rate in the mid-term and end term of model training after :code:`Adagrad` is used.

API Reference: :ref:`api_paddle_optimizer_RMSProp`


7. Adamax
------------------------
`Adamax <https://arxiv.org/abs/1412.6980>`_ is a variant of :code:`Adam` algorithm, simplifying limits of learning rate, especially upper limit.

API Reference: :ref:`api_paddle_optimizer_Adamax`


8. Lamb
------------------------
`Lamb <https://arxiv.org/abs/1904.00962>`_ aims to increase the batch size during training without compromising accuracy, supporting adaptive element-wise updates and precise layer-wise correction.

API Reference: :ref:`api_paddle_optimizer_Lamb`


9. NAdam
------------------------
`NAdam <https://openreview.net/forum?id=OM0jvwB8jIp57ZJjtNEZ>`_ optimizer is based on :code:`Adam`, combining the advantages of :code:`Nesterov` momentum and :code:`Adam`'s adaptive learning rate.

API Reference: :ref:`api_paddle_optimizer_NAdam`


10. RAdam
------------------------
`RAdam <https://arxiv.org/abs/1908.03265>`_ optimizer improves upon :code:`Adam` by introducing an adaptive learning rate warmup strategy, enhancing the initial stability of training.

API Reference: :ref:`api_paddle_optimizer_RAdam`


11. ASGD
------------------------
`ASGD <https://hal.science/hal-00860051v2>`_ optimizer, it is a strategy version of :code:`SGD` that trades space for time, and is a stochastic optimization method with trajectory averaging. On the basis of :code:`SGD`, :code:`ASGD` adds a measure of the average value of historical parameters, making the variance of noise in the descending direction decrease in a decreasing trend, so that the algorithm will eventually converge to the optimal value at a linear speed.

API Reference: :ref:`api_paddle_optimizer_ASGD`


12. Rprop
------------------------
`Rprop <https://ieeexplore.ieee.org/document/298623>`_ optimizer, this method considers that the magnitude of gradients for different weight parameters may vary greatly, making it difficult to find a global learning step size. Therefore, an innovative method is proposed to accelerate the optimization process by dynamically adjusting the learning step size through the use of parameter gradient symbols.

API Reference: :ref:`api_paddle_optimizer_Rprop`


13. LBFGS
------------------------
`LBFGS <https://en.wikipedia.org/wiki/Limited-memory_BFGS>`_ employs the limited-memory BFGS method, approximating the inverse of the Hessian matrix to update parameters.

API Reference: :ref:`api_paddle_optimizer_LBFGS`


14. Adadelta
------------------------
`Adadelta <https://arxiv.org/abs/1212.5701>`_ optimizer is an improved version of :code:`Adagrad`, using exponential moving averages to adjust the learning rate, mitigating the rapid decline in learning rate during later stages of training and enhancing stability.

API Reference: :ref:`api_paddle_optimizer_Adadelta`
