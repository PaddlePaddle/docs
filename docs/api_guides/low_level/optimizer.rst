..  _api_guide_optimizer:

###########
优化器
###########

神经网络最终是一个 `最优化问题 <https://en.wikipedia.org/wiki/Optimization_problem>`_ ，
在经过 `前向计算和反向传播 <https://zh.wikipedia.org/zh-hans/反向传播算法>`_ 后，
:code:`Optimizer` 使用反向传播梯度，优化神经网络中的参数。


1. Adam
-------------------------
`Adam <https://arxiv.org/abs/1412.6980>`_ 的优化器是一种自适应调整学习率的方法，
适用于大多非 `凸优化 <https://zh.wikipedia.org/zh/凸優化>`_ 、大数据集和高维空间的场景。在实际应用中，:code:`Adam` 是最为常用的一种优化方法。

API Reference 请参考 :ref:`cn_api_paddle_optimizer_Adam`


2. SGD
-------------------------
:code:`SGD` 是实现 `随机梯度下降 <https://arxiv.org/pdf/1609.04747>`_ 的一个 :code:`Optimizer` 子类，是 `梯度下降 <https://zh.wikipedia.org/zh-hans/梯度下降法>`_ 大类中的一种方法。
当需要训练大量样本的时候，往往选择 :code:`SGD` 来使损失函数更快的收敛。

API Reference 请参考 :ref:`cn_api_paddle_optimizer_SGD`


3. Momentum
-------------------------
:code:`Momentum` 优化器在 :code:`SGD` 基础上引入动量，减少了随机梯度下降过程中存在的噪声问题。
用户在使用时可以将 :code:`use_nesterov` 参数设置为 False 或 True ，分别对应传统 `Momentum(论文 4.1 节)
<https://arxiv.org/pdf/1609.04747>`_  算法和 `Nesterov accelerated gradient(论文 4.2 节)
<https://arxiv.org/pdf/1609.04747>`_ 算法。

API Reference 请参考 :ref:`cn_api_paddle_optimizer_Momentum`


4. AdamW
-------------------------
`AdamW <https://arxiv.org/abs/1711.05101>`_ 优化器是 :code:`Adam` 的改进版本，通过解耦权重衰减（正则化）和梯度更新，解决了 :code:`Adam` 中 L2 正则化失效的问题。

API Reference 请参考 :ref:`cn_api_paddle_optimizer_AdamW`


5. Adagrad
-------------------------
`Adagrad <http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf>`_ 优化器可以针对不同参数样本数不平均的问题，自适应地为各个参数分配不同的学习率。

API Reference 请参考 :ref:`cn_api_paddle_optimizer_Adagrad`


6. RMSProp
-------------------------
`RMSProp <http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf>`_ 优化器是一种自适应调整学习率的方法，
主要解决使用 :code:`Adagrad` 后，模型训练中后期学习率急剧下降的问题。

API Reference 请参考 :ref:`cn_api_paddle_optimizer_RMSProp`


7. Adamax
-------------------------
`Adamax <https://arxiv.org/abs/1412.6980>`_ 是 :code:`Adam` 算法的一个变种，对学习率的上限提供了一个更简单的范围，使学习率更新的算法更加稳定和简单。

API Reference 请参考 :ref:`cn_api_paddle_optimizer_Adamax`


8. Lamb
-------------------------
`Lamb <https://arxiv.org/abs/1904.00962>`_ 旨在不降低精度的前提下增大训练的批量大小，支持自适应逐元素更新和精确的分层校正。

API Reference 请参考 :ref:`cn_api_paddle_optimizer_Lamb`


9. NAdam
-------------------------
`NAdam <https://openreview.net/forum?id=OM0jvwB8jIp57ZJjtNEZ>`_ 优化器基于 :code:`Adam` 实现，结合了 :code:`Nesterov` 动量和 :code:`Adam` 自适应学习率的优点。

API Reference 请参考 :ref:`cn_api_paddle_optimizer_NAdam`


10. RAdam
-------------------------
`RAdam <https://arxiv.org/abs/1908.03265>`_ 优化器是对 :code:`Adam` 的改进，通过自适应学习率的热身 (warmup) 策略提高训练的稳定性。

API Reference 请参考 :ref:`cn_api_paddle_optimizer_RAdam`


11. ASGD
-------------------------
`ASGD <https://hal.science/hal-00860051v2>`_ 优化器，是 :code:`SGD` 以空间换时间的策略版本，是一种轨迹平均的随机优化方法。 :code:`ASGD` 在 :code:`SGD` 的基础上，增加了历史参数的平均值度量，让下降方向噪音的方差呈递减趋势下降，从而使得算法最终会以线性速度收敛于最优值。

API Reference 请参考 :ref:`cn_api_paddle_optimizer_ASGD`


12. Rprop
-------------------------
`Rprop <https://ieeexplore.ieee.org/document/298623>`_ 优化器考虑到不同权值参数的梯度的数量级可能相差很大，很难找到一个全局的学习步长。因此创新性地提出靠参数梯度的符号，动态的调节学习步长以加速优化过程的方法。

API Reference 请参考 :ref:`cn_api_paddle_optimizer_Rprop`


13. LBFGS
-------------------------
`LBFGS <https://en.wikipedia.org/wiki/Limited-memory_BFGS>`_ 采用有限内存 BFGS 方法，通过近似海森矩阵的逆更新参数。

API Reference 请参考 :ref:`cn_api_paddle_optimizer_LBFGS`


14. Adadelta
-------------------------
`Adadelta <https://arxiv.org/abs/1212.5701>`_ 优化器是 :code:`Adagrad` 的改进版本，通过指数移动平均调整学习率，缓解 :code:`Adagrad` 后期学习率过快下降的问题，提升训练稳定性。

API Reference 请参考 :ref:`cn_api_paddle_optimizer_Adadelta`
