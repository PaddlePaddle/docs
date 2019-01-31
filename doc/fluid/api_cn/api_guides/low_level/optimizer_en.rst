..  _api_guide_optimizer_en:

###########
Optimizer
###########

Neural network finally is a `Optimization problem <https://en.wikipedia.org/wiki/Optimization_problem>`_ ,
With `forward computing and backpropagation <https://zh.wikipedia.org/zh-hans/backpropagation_algorithm>`_ ,
:code:`Optimizer` use backpropagation gradient to optimize parameters in neural network.

1.SGD/SGDOptimizer
------------------

:code:`SGD` is a child class of :code:`Optimizer` implementing `Radom Gradient Descent <https://arxiv.org/pdf/1609.04747.pdf>`_ as well as a method of `Gradient Descent <https://zh.wikipedia.org/zh-hans/gradient_descent_algorithm>`_ .
When it needs to train a large number of samples, we usually choose :code:`SGD` to make loss function converge more quickly.  

About API Reference, please refer to :ref:`api_fluid_optimizer_SGDOptimizer`


2.Momentum/MomentumOptimizer
----------------------------

:code:`Momentum` optimizer adds momentum on the basis of :code:`SGD` , reducing noise problem in the process of random gradient descent.
You can set :code:`ues_nesterov` as False or True, repectively correspondent with traditional `Momentum(Section 4.1 in thesis)
<https://arxiv.org/pdf/1609.04747.pdf>`_  algorithm and `Nesterov accelerated gradient(Section 4.2 in thesis)
<https://arxiv.org/pdf/1609.04747.pdf>`_ algorithm.

About API Reference, please refer to :ref:`api_fluid_optimizer_MomentumOptimizer`


3. Adagrad/AdagradOptimizer
---------------------------
`Adagrad <http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf>`_ Optimizer can adaptively allocate different learning rates for parameters to solve the problem of for different sample sizes for different parameters.

About API Reference, please refer to :ref:`api_fluid_optimizer_AdagradOptimizer`


4.RMSPropOptimizer
------------------
`RMSProp optimizer <http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf>`_  is a method adaptively adjusting learning rate.
It mainly solves the problem of dramatic decrease of learning rate in the mid-term and end term of model training.

About API Reference, please refer to :ref:`api_fluid_optimizer_RMSPropOptimizer`



5.Adam/AdamOptimizer
--------------------
Optimizer of `Adam <https://arxiv.org/abs/1412.6980>`_ is a method to adaptively adjust learning rate,
fitting for most beyond `convex optimization <https://zh.wikipedia.org/zh/convex_optimization>`_ , big data set and real scenarios. :code:`Adam` is the most common optimization algorithm.

About API Reference, please refer to :ref:`api_fluid_optimizer_AdamOptimizer`



6.Adamax/AdamaxOptimizer
------------------------

`Adamax <https://arxiv.org/abs/1412.6980>`_ is a variant of :code:`Adam` algorithm, simplifying limit of learning rate, especially upper limit.the ceiling range of learning rate .

About API Reference, please refer to :ref:`api_fluid_optimizer_AdamaxOptimizer`



7.DecayedAdagrad/ DecayedAdagradOptimizer
-------------------------------------------

`DecayedAdagrad <http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf>`_ optimizer can be regarded as :code:`Adagrad` algorithm added with decay rate to solve the problem of dramatic descent of learning rate in mid-term and end term in model training.

About API Reference, please refer to :ref:`api_fluid_optimizer_DecayedAdagrad`




8. Ftrl/FtrlOptimizer
----------------------

`FtrlOptimizer <https://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf>`_ optimizer combines hign accuracy of `FOBOS algorithm <https://stanford.edu/~jduchi/projects/DuchiSi09b.pdf>`_ and sparse of `RDA algorithm
<http://www1.se.cuhk.edu.hk/~sqma/SEEM5121_Spring2015/dual-averaging.pdf>`_ , which is an algorithm `Online Learning <https://en.wikipedia.org/wiki/Online_machine_learning>`_ with extremely good effect.

About API Reference, please refer to :ref:`api_fluid_optimizer_FtrlOptimizer`



9.ModelAverage
-----------------

:code:`ModelAverage` optimizer accumulates history parameter with window during the modeling.We use average paramet at inference to upgrade accuracy of inference.
About API Reference, please refer to :ref:`api_fluid_optimizer_ModelAverage`

