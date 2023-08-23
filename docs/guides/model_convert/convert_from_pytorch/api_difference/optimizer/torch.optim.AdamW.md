## [ torch 参数更多 ]torch.optim.AdamW

### [torch.optim.AdamW](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html)

```python
torch.optim.AdamW(params,
                lr=0.001,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=0.01,
                amsgrad=False,
                maximize=False,
                foreach=None,
                capturable=False,
                differentiable=False,
                fused=None)
```

### [paddle.optimizer.AdamW](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/optimizer/AdamW_cn.html)

```python
paddle.optimizer.AdamW(learning_rate=0.001,
                    beta1=0.9,
                    beta2=0.999,
                    epsilon=1e-08,
                    parameters=None,
                    weight_decay=0.01,
                    lr_ratio=None,
                    apply_decay_param_fun=None,
                    grad_clip=None,
                    name=None,
                    lazy_mode=False,
                    multi_precision=False,)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch                             | PaddlePaddle | 备注                                                                    |
| ----------------------------------- | ------------ | ----------------------------------------------------------------------- |
| params     | parameters           | 表示指定优化器需要优化的参数，仅参数名不一致。                      |
| lr     | learning_rate       | 学习率，用于参数更新的计算。仅参数名不一致。                          |
| betas     | beta1、beta2       | 一阶矩估计的指数衰减率。Pytorch 为元祖形式，Paddle 为分开的两个参数。默认值分别一致。                          |
| eps       | epsilon        | 保持数值稳定性的短浮点类型值。仅参数名不一致。                           |
| weight_decay           | weight_decay     | 表示权重衰减系数。参数名和默认值均一致。         |
| amsgrad   | -    | 是否使用该算法的 AMSGrad 变体。Paddle 无此参数，暂无转写方式。                       |
| maximize           | -     | 根据目标最大化参数，而不是最小化。Paddle 无此参数，暂无转写方式。         |
| foreach           | -     | 是否使用优化器的 foreach 实现。Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。         |
| capturable           | -     | 在 CUDA 图中捕获此实例是否安全。Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。         |
| differentiable      | -     | 是否应通过训练中的优化器步骤进行自动微分。Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。    |
| fused      | -     | 是否使用融合实现（仅限 CUDA）。Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。       |
| -          | lr_ratio            | 传入函数时，会为每个参数计算一个权重衰减系数，并使用该系数与学习率的乘积作为新的学习率。PyTorch 无此参数，Paddle 保持默认即可。       |
| -          | apply_decay_param_fun            | 传入函数时，只有可以使 apply_decay_param_fun(Tensor.name)==True 的 Tensor 会进行 weight decay 更新。PyTorch 无此参数，Paddle 保持默认即可。       |
| -          | grad_clip            | 梯度裁剪的策略。 PyTorch 无此参数，Paddle 保持默认即可。       |
| -          | lazy_mode            | 设为 True 时，仅更新当前具有梯度的元素。PyTorch 无此参数，Paddle 保持默认即可。       |
| -          | multi_precision      |  在基于 GPU 设备的混合精度训练场景中，该参数主要用于保证梯度更新的数值稳定性。PyTorch 无此参数，Paddle 保持默认即可。       |
