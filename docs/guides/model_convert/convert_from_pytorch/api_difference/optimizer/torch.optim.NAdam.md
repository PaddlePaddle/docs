## [ torch 参数更多 ]torch.optim.NAdam

### [torch.optim.NAdam](https://pytorch.org/docs/stable/generated/torch.optim.NAdam.html#torch.optim.NAdam)

```python
torch.optim.NAdam(params,
                lr=0.002,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=0,
                momentum_decay=0.004,
                decoupled_weight_decay=False,
                foreach=None,
                maximize=False,
                capturable=False,
                differentiable=False)
```

### [paddle.optimizer.NAdam](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/optimizer/NAdam_cn.html#nadam)

```python
paddle.optimizer.NAdam(learning_rate=0.002,
                    beta1=0.9,
                    beta2=0.999,
                    epsilon=1e-08,
                    momentum_decay=0.004,
                    parameters=None,
                    weight_decay=None,
                    grad_clip=None,
                    name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch                             | PaddlePaddle | 备注                                                                    |
| ----------------------------------- | ------------ | ----------------------------------------------------------------------- |
| params     | parameters           | 表示指定优化器需要优化的参数，仅参数名不一致。                      |
| lr     | learning_rate       | 学习率，用于参数更新的计算。仅参数名不一致。                          |
| betas     | beta1, beta2       | 一阶矩估计的指数衰减率。PyTorch 为元祖形式，Paddle 为分开的两个参数。默认值分别一致。                          |
| eps       | epsilon        | 保持数值稳定性的短浮点类型值。仅参数名不一致。                           |
| weight_decay           | weight_decay     | 表示权重衰减系数，参数默认值不一致, PyTorch 默认为`0`， Paddle 默认为`None`，Paddle 需保持与 PyTorch 一致。         |
| momentum_decay | momentum_decay | 动量衰减率。|
| decoupled_weight_decay   | -    | 是否像 AdamW 那样使用解耦权重衰减。Paddle 无此参数，暂无转写方式。                       |
| foreach           | -     | 是否使用优化器的 foreach 实现。Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。         |
| maximize           | -     | 根据目标最大化参数，而不是最小化。Paddle 无此参数，暂无转写方式。         |
| capturable           | -     | 在 CUDA 图中捕获此实例是否安全。Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。         |
| differentiable      | -     | 是否应通过训练中的优化器步骤进行自动微分。Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。         |
| -          | grad_clip            | 梯度裁剪的策略。 PyTorch 无此参数，Paddle 保持默认即可。       |
