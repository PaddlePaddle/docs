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
sgd = torch.optim.SGD(
    params=linear.parameters(),
    lr=0.1,
    weight_decay=0.01
)
sgd.step()

# Paddle 写法
sgd = paddle.optimizer.SGD(
     learning_rate=0.1,
     parameters=linear.parameters(),
     weight_decay=0.01
)
sgd.step()
```
