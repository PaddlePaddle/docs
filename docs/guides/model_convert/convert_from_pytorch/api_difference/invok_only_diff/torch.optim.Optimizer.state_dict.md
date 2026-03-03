## [ 仅 API 调用方式不一致 ]torch.optim.Optimizer.state_dict

### [torch.optim.Optimizer.state\_dict](https://docs.pytorch.org/docs/stable/generated/torch.optim.Optimizer.state_dict.html#torch.optim.Optimizer.state_dict)

```python
torch.optim.Optimizer.state_dict()
```

### [paddle.optimizer.Optimizer.state_dict](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/optimizer/Optimizer_cn.html#state-dict)

```python
paddle.optimizer.Optimizer.state_dict()
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
optim = torch.optim.Optimizer([theta], defaults={"learning_rate": 1.0})
result = optim.state_dict()

# Paddle 写法
optim = paddle.optimizer.Optimizer(parameters=[theta], **{"learning_rate": 1.0})
result = optim.state_dict()
```
