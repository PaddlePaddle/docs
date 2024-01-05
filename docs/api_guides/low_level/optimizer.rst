..  _api_guide_optimizer:

###########
优化器
###########

神经网络最终是一个 `最优化问题 <https://en.wikipedia.org/wiki/Optimization_problem>`_ ，
在经过 `前向计算和反向传播 <https://zh.wikipedia.org/zh-hans/反向传播算法>`_ 后，
:code:`Optimizer` 使用反向传播梯度，优化神经网络中的参数。

1.SGD/SGDOptimizer
------------------

:code:`SGD` 是实现 `随机梯度下降 <https://arxiv.org/pdf/1609.04747.pdf>`_ 的一个 :code:`Optimizer` 子类，是 `梯度下降 <https://zh.wikipedia.org/zh-hans/梯度下降法>`_ 大类中的一种方法。
当需要训练大量样本的时候，往往选择 :code:`SGD` 来使损失函数更快的收敛。

API Reference 请参考 :ref:`cn_api_fluid_optimizer_SGDOptimizer`


2.Momentum/MomentumOptimizer
----------------------------

:code:`Momentum` 优化器在 :code:`SGD` 基础上引入动量，减少了随机梯度下降过程中存在的噪声问题。
用户在使用时可以将 :code:`ues_nesterov` 参数设置为 False 或 True，分别对应传统 `Momentum(论文 4.1 节)
<https://arxiv.org/pdf/1609.04747.pdf>`_  算法和 `Nesterov accelerated gradient(论文 4.2 节)
<https://arxiv.org/pdf/1609.04747.pdf>`_ 算法。

API Reference 请参考 :ref:`cn_api_fluid_optimizer_MomentumOptimizer`


3. Adagrad/AdagradOptimizer
---------------------------
`Adagrad <http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf>`_ 优化器可以针对不同参数样本数不平均的问题，自适应地为各个参数分配不同的学习率。

API Reference 请参考 :ref:`cn_api_fluid_optimizer_AdagradOptimizer`


4.RMSPropOptimizer
------------------
`RMSProp 优化器 <http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf>`_ ，是一种自适应调整学习率的方法，
主要解决使用 Adagrad 后，模型训练中后期学习率急剧下降的问题。

API Reference 请参考 :ref:`cn_api_fluid_optimizer_RMSPropOptimizer`



5.Adam/AdamOptimizer
--------------------
`Adam <https://arxiv.org/abs/1412.6980>`_ 的优化器是一种自适应调整学习率的方法，
适用于大多非 `凸优化 <https://zh.wikipedia.org/zh/凸優化>`_ 、大数据集和高维空间的场景。在实际应用中，:code:`Adam` 是最为常用的一种优化方法。

API Reference 请参考 :ref:`cn_api_fluid_optimizer_AdamOptimizer`



6.Adamax/AdamaxOptimizer
------------------------

`Adamax <https://arxiv.org/abs/1412.6980>`_ 是 :code:`Adam` 算法的一个变体，对学习率的上限提供了一个更简单的范围，使学习率的边界范围更简单。

API Reference 请参考 :ref:`cn_api_fluid_optimizer_AdamaxOptimizer`



7.DecayedAdagrad/ DecayedAdagradOptimizer
-------------------------------------------

`DecayedAdagrad <http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf>`_ 优化器，可以看做是引入了衰减速率的 :code:`Adagrad` 算法，解决使用 Adagrad 后，模型训练中后期学习率急剧下降的问题。

API Reference 请参考 :ref:`cn_api_fluid_optimizer_DecayedAdagrad`




8. Ftrl/FtrlOptimizer
----------------------

`FtrlOptimizer <https://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf>`_ 优化器结合了 `FOBOS 算法 <https://stanford.edu/~jduchi/projects/DuchiSi09b.pdf>`_ 的高精度与 `RDA 算法
<http://xueshu.baidu.com/usercenter/paper/show?paperid=101df241a792fe23d79f4ed84a820495>`_ 的稀疏性，是目前效果非常好的一种 `Online Learning <https://en.wikipedia.org/wiki/Online_machine_learning>`_ 算法。

API Reference 请参考 :ref:`cn_api_fluid_optimizer_FtrlOptimizer`



9.ModelAverage
-----------------

:code:`ModelAverage` 优化器，在训练中通过窗口来累计历史 parameter，在预测时使用取平均值后的 paramet，整体提高预测的精度。

API Reference 请参考 :ref:`cn_api_fluid_optimizer_ModelAverage`




10.Rprop/RpropOptimizer
-----------------

:code:`Rprop` 优化器，该方法考虑到不同权值参数的梯度的数量级可能相差很大，因此很难找到一个全局的学习步长。因此创新性地提出靠参数梯度的符号，动态的调节学习步长以加速优化过程的方法。

API Reference 请参考 :ref:`cn_api_fluid_optimizer_Rprop`