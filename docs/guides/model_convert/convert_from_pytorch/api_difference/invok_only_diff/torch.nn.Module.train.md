## [ 仅 API 调用方式不一致 ]torch.nn.Module.train

### [torch.nn.Module.train](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.train)

```python
torch.nn.Module.train(mode=True)
```

### [paddle.nn.Layer.train](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/Layer/train_cn.html#paddle/nn/Layer/train_cn#cn-api-paddle-nn-Layer-train)

```python
paddle.nn.Layer.train()
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
model = torch.nn.Linear(10, 20)
model.train()

# Paddle 写法
model = paddle.nn.Linear(10, 20)
model.train()
```
