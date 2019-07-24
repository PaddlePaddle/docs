.. _cn_api_fluid_optimizer_DGCMomentumOptimizer:

DGCMomentumOptimizer
-------------------------------

.. py:class:: paddle.fluid.optimizer.DGCMomentumOptimizer(learning_rate, momentum, rampup_begin_step, rampup_step=1, sparsity=[0.999], use_nesterov=False, local_grad_clip_norm=None, num_trainers=None, regularization=None, name=None)

原始论文: https://arxiv.org/abs/1712.01887

DGC通过仅发送重要梯度（稀疏更新）来减少通信带宽：仅发送大于给定阈值的梯度。

为避免丢失信息，DGC在本地累积其余梯度。最终，这些梯度会积累到足够大，从而可以传输。

因此，DGC即时发送相对较大的梯度，但最终随时间积累而发送所有梯度。

此外，为了确保不损失精度，DGC在梯度稀疏化之上采用动量修正和局部梯度修剪(clip)来维持模型性能。

DGC还使用动量因子掩藏(momentum factor masking)和预训练(warm-up)来克服由于reduced通讯而导致的数据陈旧性(staleness)问题。

这个优化器会执行如下操作：

1. 通过从张量获取前TopK个导入值来压缩梯度，并将其用于allreduce以减少网络带宽。
2. 调用momentum来降低cost。

参数: 
    - **learning_rate** （float | Variable） - 用于更新参数的学习率。可以是浮点值或由一个浮点型数据组成的Variable。
    - **momentum** （float） - 动量因子。
    - **rampup_begin_step** （int） - 进行梯度压缩的起步点。
    - **rampup_step** （int） - 使用稀疏期的时间。默认值为1.例如：如果稀疏度为[0.75,0.9375,0.984375,0.996,0.999]，并且rampup_step为5，则在0步时使用0.75，在1步时使用0.9375，依此类推。当达到sparsity数组末尾时，它此后延续使用0.999。
    - **sparsity** （list [float]） - 从梯度张量中获取较为重要的元素，比率为（1-当前稀疏度）。
    - **use_nesterov** （bool） - 启用Nesterov momentum。 True意味着使用nesterov。
    - **local_grad_clip_norm** （float） - 如果需要，clip norm值。
    - **num_trainers**   - 训练节点的数量。
    - **regularization**  - 正则器，如fluid.regularizer.L2DecayRegularizer。
    - **name**   - 可选的名称前缀。

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    optimizer = fluid.optimizer.DGCMomentumOptimizer(
                                        learning_rate=0.0001,
                                        momentum=0.9,
                                        rampup_step=1000,
                                        rampup_begin_step=1252,
                                        sparsity=[0.999, 0.999])



