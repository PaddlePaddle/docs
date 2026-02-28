## [ 仅 API 调用方式不一致 ]torch.optim.Optimizer.load_state_dict

### [torch.optim.Optimizer.load\_state\_dict](https://docs.pytorch.org/docs/stable/generated/torch.optim.Optimizer.load_state_dict.html#torch.optim.Optimizer.load_state_dict)

```python
torch.optim.Optimizer.load_state_dict(state_dict)
```

### [paddle.optimizer.Optimizer.load_state_dict](https://www.paddlepaddle.org.cn/documentation/docs/en/develop/api/paddle/optimizer/Optimizer_en.html#set_state_dict)

```python
paddle.optimizer.Optimizer.load_state_dict(state_dict)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
optim = torch.optim.SGD([theta], lr=1.0)
result = optim.state_dict()
optim.load_state_dict(result)

# Paddle 写法
optim = paddle.optimizer.SGD(learning_rate=1.0, parameters=[theta])
result = optim.state_dict()
optim.load_state_dict(result)

```
