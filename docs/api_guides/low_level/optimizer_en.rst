..  _api_guide_optimizer_en:

###########
Optimizer
###########

Neural network in essence is a `Optimization problem <https://en.wikipedia.org/wiki/Optimization_problem>`_ .
With `forward computing and back propagation <https://zh.wikipedia.org/zh-hans/backpropagation_algorithm>`_ ,
:code:`Optimizer` use back-propagation gradients to optimize parameters in a neural network.

1.SGD/SGDOptimizer
------------------

:code:`SGD` is an offspring class of :code:`Optimizer` implementing `Random Gradient Descent <https://arxiv.org/pdf/1609.04747.pdf>`_ which is a method of `Gradient Descent <https://zh.wikipedia.org/zh-hans/gradient_descent_algorithm>`_ .
When it needs to train a large number of samples, we usually choose :code:`SGD` to make loss function converge more quickly.

API Reference: :ref:`api_fluid_optimizer_SGDOptimizer`


2.Momentum/MomentumOptimizer
----------------------------

:code:`Momentum` optimizer adds momentum on the basis of :code:`SGD` , reducing noise problem in the process of random gradient descent.
You can set :code:`ues_nesterov` as False or True, respectively corresponding to traditional `Momentum(Section 4.1 in thesis)
<https://arxiv.org/pdf/1609.04747.pdf>`_  algorithm and `Nesterov accelerated gradient(Section 4.2 in thesis)
<https://arxiv.org/pdf/1609.04747.pdf>`_ algorithm.

API Reference: :ref:`api_fluid_optimizer_MomentumOptimizer`


3. Adagrad/AdagradOptimizer
---------------------------
`Adagrad <http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf>`_ Optimizer can adaptively allocate different learning rates for parameters to solve the problem of different sample sizes for different parameters.

API Reference: :ref:`api_fluid_optimizer_AdagradOptimizer`


4.RMSPropOptimizer
------------------
`RMSProp optimizer <http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf>`_  is a method to adaptively adjust learning rate.
It mainly solves the problem of dramatic decrease of learning rate in the mid-term and end term of model training after Adagrad is used.

API Reference: :ref:`api_fluid_optimizer_RMSPropOptimizer`



5.Adam/AdamOptimizer
--------------------
Optimizer of `Adam <https://arxiv.org/abs/1412.6980>`_ is a method to adaptively adjust learning rate,
fit for most  non- `convex optimization <https://zh.wikipedia.org/zh/convex_optimization>`_ , big data set and high-dimensional scenarios. :code:`Adam` is the most common optimization algorithm.

API Reference: :ref:`api_fluid_optimizer_AdamOptimizer`



6.Adamax/AdamaxOptimizer
------------------------

`Adamax <https://arxiv.org/abs/1412.6980>`_ is a variant of :code:`Adam` algorithm, simplifying limits of learning rate, especially upper limit.

API Reference: :ref:`api_fluid_optimizer_AdamaxOptimizer`



7.DecayedAdagrad/DecayedAdagradOptimizer
-------------------------------------------

`DecayedAdagrad <http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf>`_ Optimizer can be regarded as an :code:`Adagrad` algorithm incorporated with decay rate to solve the problem of dramatic descent of learning rate in mid-term and end term of model training.

API Reference: :ref:`api_fluid_optimizer_DecayedAdagrad`




8. Ftrl/FtrlOptimizer
----------------------

`FtrlOptimizer <https://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf>`_ Optimizer combines the high accuracy of `FOBOS algorithm <https://stanford.edu/~jduchi/projects/DuchiSi09b.pdf>`_ and the sparsity of `RDA algorithm <http://www1.se.cuhk.edu.hk/~sqma/SEEM5121_Spring2015/dual-averaging.pdf>`_ , which is an `Online Learning <https://en.wikipedia.org/wiki/Online_machine_learning>`_ algorithm with significantly satisfying effect.

API Reference: :ref:`api_fluid_optimizer_FtrlOptimizer`



9.ModelAverage
-----------------

:code:`ModelAverage` Optimizer accumulates history parameters through sliding window during the model training. We use averaged parameters at inference time to upgrade general accuracy of inference.

API Reference: :ref:`api_fluid_optimizer_ModelAverage`




10.Rprop/RpropOptimizer
-----------------

:code:`Rprop` Optimizer, this method considers that the magnitude of gradients for different weight parameters may vary greatly, making it difficult to find a global learning step size. Therefore, an innovative method is proposed to accelerate the optimization process by dynamically adjusting the learning step size through the use of parameter gradient symbols.

API Reference: :ref:`api_fluid_optimizer_Rprop`




11.ASGD/ASGDOptimizer
-----------------

:code:`ASGD` Optimizer, it is a strategy version of SGD that trades space for time, and is a stochastic optimization method with trajectory averaging. On the basis of SGD, ASGD adds a measure of the average value of historical parameters, making the variance of noise in the descending direction decrease in a decreasing trend, so that the algorithm will eventually converge to the optimal value at a linear speed.

API Reference: :ref:`api_fluid_optimizer_ASGD`
