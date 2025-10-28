## [ 仅 API 调用方式不一致 ]torch.nn.Module.modules

### [torch.nn.Module.modules](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.modules)

```python
torch.nn.Module.modules()
```

### [paddle.nn.Layer.sublayers](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/Layer/sublayers_cn.html#paddle/nn/Layer/sublayers_cn#cn-api-paddle-nn-Layer-sublayers)

```python
paddle.nn.Layer.sublayers(include_self=False)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
model = torch.nn.Sequential(net1, net2)
for layer in model.modules():
    print(layer)

# Paddle 写法
model = paddle.nn.Sequential(net1, net2)
for layer in model.sublayers():
    print(layer)
```
