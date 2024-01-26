## [ torch 参数更多 ]torch.optim.ASGD

### [torch.optim.ASGD](https://pytorch.org/docs/stable/generated/torch.optim.ASGD.html)

```python
torch.optim.ASGD(params,
                lr=0.01, 
                lambd=0.0001, 
                alpha=0.75, 
                t0=1000000.0, 
                weight_decay=0, 
                foreach=None, 
                maximize=False, 
                differentiable=False)
```

### [paddle.optimizer.ASGD](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/optimizer/ASGD_cn.html#cn-api-paddle-optimizer-asgd)

```python
paddle.optimizer.ASGD(learning_rate=0.001,
                    batch_num=1,
                    parameters=None,
                    weight_decay=None,
                    grad_clip=None,
                    multi_precision=False,
                    name=None)
```

注：Pytorch 的 ASGD 是有问题的。
Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle        | 备注                                                                                                                    |
| ------------- | ------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| params        | parameters          | 表示指定优化器需要优化的参数，仅参数名不一致                                                                                |
| lr            | learning_rate       | 学习率，用于参数更新的计算。参数默认值不一致, Pytorch 默认为 `0.0001`， Paddle 默认为 `0.001`，Paddle 需保持与 Pytorch 一致    |
| lambd         | -                   | 衰变项，与 weight_decay 功能重叠，可直接删除                                                                               |
| alpha         | -                   | eta 更新的 power，可直接删除                                                                                              |
| t0            | -                   | 开始求平均值的点，可直接删除                                                                                               |
| weight_decay  | weight_decay        | 权重衰减。参数默认值不一致, Pytorch 默认为 `0`， Paddle 默认为 `None`，Paddle 需保持与 Pytorch 一致                           |
| foreach       | -                   | 是否使用优化器的 foreach 实现。Paddle 无此参数，一般对网络训练结果影响不大，可直接删除                                         |
| maximize      | -                   | 根据目标最大化参数，而不是最小化。Paddle 无此参数，暂无转写方式                                                               |
| differentiable| -                   | 是否应通过训练中的优化器步骤进行自动微分。Paddle 无此参数，一般对网络训练结果影响不大，可直接删除                                |
| -             | batch_num           | 完成一个 epoch 所需迭代的次数。 PyTorch 无此参数。假设样本总数为 all_size，Paddle 需将 batch_num 设置为 all_size / batch_size |
| -             | grad_clip           | 梯度裁剪的策略。 PyTorch 无此参数，Paddle 保持默认即可                                                                      |
| -             | multi_precision     | 在基于 GPU 设备的混合精度训练场景中，该参数主要用于保证梯度更新的数值稳定性。 PyTorch 无此参数，Paddle 保持默认即可              |
| -             | name                | 一般情况下无需设置。 PyTorch 无此参数，Paddle 保持默认即可                                                                  |

### 相关问题

torch 当前版本的 ASGD 实现并不完善。转换过来的 paddle ASGD 会与 torch 的不一致（不影响收敛），但是可以正常使用。如果强需求保证转换前后一致，可以自行尝试其他优化器。

如果后续 torch 有代码更新，可以联系 @WintersMontagne10335 作 API 调整与对接。

#### torch 现存问题

在 `_single_tensor_asgd` 中，对 `axs, ax` 进行了更新，但是它们却并没有参与到 `params` 中。 `axs, ax` 完全没有作用。

调研到的比较可信的原因是，目前 `ASGD` 的功能并不完善， `axs, ax` 是预留给以后的版本的。

另外，weight_decay 是冗余的。

当前版本 `ASGD` 的功能，类似于 `SGD` 。

详情可见：
- https://discuss.pytorch.org/t/asgd-optimizer-has-a-bug/95060
- https://discuss.pytorch.org/t/averaged-sgd-implementation/26960
- https://github.com/pytorch/pytorch/issues/74884

#### paddle 实现思路

主要参照 [`ASGD` 论文: Minimizing Finite Sums with the Stochastic Average Gradient](https://inria.hal.science/hal-00860051v2)

核心步骤为：

    1. 初始化 d, y
    2. 随机采样
    3. 用本次计算得到的第 i 个样本的梯度信息，替换上一次的梯度信息
    4. 更新参数

伪代码和详细实现步骤可见：
- https://github.com/PaddlePaddle/community/blob/b76313c3b8f8b6a2f808d90fa95dcf265dbef67d/rfcs/APIs/20231111_api_design_for_ASGD.md
