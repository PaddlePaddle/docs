## [ torch 参数更多 ]torch.optim.Optimizer.step

### [torch.optim.Optimizer.step](https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.html#torch.optim.Optimizer.step)

```python
torch.optim.Optimizer.step(closure=None)
```

### [paddle.optimizer.Optimizer.step](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/optimizer/Optimizer/step_cn.html#paddle/optimizer/Optimizer/step_cn#cn-api-paddle-optimizer-Optimizer-step)

```python
paddle.optimizer.Optimizer.step()
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch    | PaddlePaddle | 备注 |
| ---------- | ------------ | -- |
| closure    | -           | 传入一个闭包函数，在优化器更新参数前执行。Paddle 无此参数，暂无转写方式。|
