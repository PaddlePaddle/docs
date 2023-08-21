## [ torch 参数更多 ]torch.optim.Adam

### [torch.optim.Adam](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html)

```python
torch.optim.Adam(params,
                lr=0.001,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=0,
                amsgrad=False,
                maximize=False,
                capturable=False,
                differentiable=False,
                fused=None)
```

### [paddle.optimizer.Adam](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/optimizer/Adam_cn.html)

```python
paddle.optimizer.Adam(learning_rate=0.001,
                    beta1=0.9,
                    beta2=0.999,
                    epsilon=1e-08,
                    parameters=None,
                    weight_decay=None,
                    grad_clip=None,
                    lazy_mode=False,
                    multi_precision=False,
                    use_multi_tensor=False,
                    name=None)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch                             | PaddlePaddle | 备注                                                                    |
| ----------------------------------- | ------------ | ----------------------------------------------------------------------- |
| params     | parameters           | 表示指定优化器需要优化的参数，仅参数名不一致。                      |
| lr     | learning_rate       | 学习率，用于参数更新的计算。仅参数名不一致。                          |
| betas     | beta1、beta2       | 一阶矩估计的指数衰减率。Pytorch 为元祖形式，Paddle 为分开的两个参数。默认值分别一致。                          |
| eps       | epsilon        | 保持数值稳定性的短浮点类型值。仅参数名不一致。                           |
| weight_decay           | weight_decay     | 表示权重衰减系数，参数默认值不一致, Pytorch 默认为`0`， Paddle 默认为`None`，Paddle 需保持与 Pytorch 一致。         |
| amsgrad   | -    | 是否使用该算法的 AMSGrad 变体。Paddle 无此参数，暂无转写方式。                       |
| maximize           | -     | 根据目标最大化参数，而不是最小化。Paddle 无此参数，暂无转写方式。         |
| capturable           | -     | 在 CUDA 图中捕获此实例是否安全。Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。         |
| differentiable      | -     | 是否应通过训练中的优化器步骤进行自动微分。Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。         |
| fused      | -     | 是否使用融合实现（仅限 CUDA）。Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。         |
| -          | grad_clip            | 梯度裁剪的策略。 PyTorch 无此参数，Paddle 保持默认即可。       |
| -          | lazy_mode            | 设为 True 时，仅更新当前具有梯度的元素。PyTorch 无此参数，Paddle 保持默认即可。       |
| -          | multi_precision            | 是否在权重更新期间使用 multi-precision。PyTorch 无此参数，Paddle 保持默认即可。       |
| -          | use_multi_tensor            | 是否使用 multi-tensor 策略一次性更新所有参数。PyTorch 无此参数，Paddle 保持默认即可。       |
