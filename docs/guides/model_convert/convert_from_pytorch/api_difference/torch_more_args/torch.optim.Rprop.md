## [ torch 参数更多 ]torch.optim.Rprop

### [torch.optim.Rprop](https://pytorch.org/docs/stable/generated/torch.optim.Rprop.html)

```python
torch.optim.Rprop(params,
             lr=0.01,
             etas=(0.5, 1.2),
             step_sizes=(1e-06, 50),
             foreach=None,
             maximize=False,
             differentiable=False)
```

### [paddle.optimizer.Rprop](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/optimizer/Rprop_cn.html#cn-api-paddle-optimizer-rprop)

```python
paddle.optimizer.Rprop(learning_rate=0.001,
                          learning_rate_range=(1e-5, 50),
                          parameters=None,
                          etas=(0.5, 1.2),
                          grad_clip=None,
                          name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle        | 备注                                                                                                                    |
| ------------- | ------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| params        | parameters          | 表示指定优化器需要优化的参数，仅参数名不一致。                                                                              |
| lr            | learning_rate       | 初始学习率，用于参数更新的计算。参数默认值不一致, PyTorch 默认为`0.01`， Paddle 默认为`0.001`，Paddle 需保持与 PyTorch 一致。  |
| etas          | etas                | 用于更新学习率。参数一致。                                                                                                |
| step_sizes    | learning_rate_range | 学习率的范围，参数默认值不一致, PyTorch 默认为`(1e-06, 50)`， Paddle 默认为`(1e-5, 50)`，Paddle 需保持与 PyTorch 一致。      |
| foreach       | -                   | 是否使用优化器的 foreach 实现。Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。                                      |
| maximize      | -                   | 根据目标最大化参数，而不是最小化。Paddle 无此参数，暂无转写方式。                                                            |
| differentiable| -                   | 是否应通过训练中的优化器步骤进行自动微分。Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。                             |
| -             | grad_clip           | 梯度裁剪的策略。 PyTorch 无此参数，Paddle 保持默认即可。                                                                   |
