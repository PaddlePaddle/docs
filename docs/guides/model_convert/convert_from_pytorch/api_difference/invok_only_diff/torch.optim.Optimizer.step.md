## [ 仅 API 调用方式不一致 ]torch.optim.Optimizer.step

### [torch.optim.Optimizer.step](https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.html#torch.optim.Optimizer.step)

```python
torch.optim.Optimizer.step(closure=None)
```

### [paddle.optimizer.Optimizer.step](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/optimizer/Optimizer/step_cn.html#paddle/optimizer/Optimizer/step_cn#cn-api-paddle-optimizer-Optimizer-step)

```python
paddle.optimizer.Optimizer.step()
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
optim = torch.optim.Optimizer([theta], defaults={"learning_rate": 1.0})
result = type(optim.step)
# Paddle 写法
optim = paddle.optimizer.Optimizer(parameters=[theta], **{"learning_rate": 1.0})
result = type(optim.step)
```
