## [ 仅 paddle 参数更多 ]torch.optim.Optimizer

### [torch.optim.Optimizer](https://pytorch.org/docs/stable/optim.html#torch.optim.Optimizer)

```python
torch.optim.Optimizer(params,
                    defaults)
```

### [paddle.optimizer.Optimizer](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/optimizer/Optimizer_cn.html)

```python
paddle.optimizer.Optimizer(learning_rate=0.001,
                        epsilon=1e-08,
                        parameters=None,
                        weight_decay=None,
                        grad_clip=None,
                        name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，其 `defaults` 可以支持各种参数，但一般只会转写 API 名称，不会转写参数。

### 参数映射

| PyTorch                             | PaddlePaddle | 备注                                                                    |
| ----------------------------------- | ------------ | ----------------------------------------------------------------------- |
| params     | parameters           | 表示指定优化器需要优化的参数，仅参数名不一致。                      |
| defaults     | -     | 表示含有优化选项和其默认值的字典，Paddle 无此参数。                      |
| -     | learning_rate       | 学习率，用于参数更新的计算。PyTorch 无此参数，但 defaults 可含有 lr 与之对应。                          |
| -     | weight_decay      | 表示权重衰减系数。 PyTorch 无此参数，但 defaults 可含有 weight_decay 与之对应。             |
| -      | epsilon        | 保持数值稳定性的短浮点类型值。PyTorch 无此参数，但 defaults 可含有 eps 与之对应。                           |
| -          | grad_clip            | 梯度裁剪的策略。 PyTorch 无此参数，Paddle 保持默认即可。       |
