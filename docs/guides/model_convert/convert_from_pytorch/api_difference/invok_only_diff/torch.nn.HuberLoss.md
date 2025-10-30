## [ 仅 API 调用方式不一致 ]torch.nn.HuberLoss

### [torch.nn.HuberLoss](https://pytorch.org/docs/stable/generated/torch.nn.HuberLoss.html#torch.nn.HuberLoss)

```python
torch.nn.HuberLoss(reduction='mean', delta=1.0)
```

### [paddle.nn.SmoothL1Loss](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/SmoothL1Loss_cn.html#paddle/nn/SmoothL1Loss_cn#cn-api-paddle-nn-SmoothL1Loss)

```python
paddle.nn.SmoothL1Loss(reduction='mean', delta=1.0, name=None)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
loss = torch.nn.HuberLoss()

# Paddle 写法
loss = paddle.nn.SmoothL1Loss()
```
