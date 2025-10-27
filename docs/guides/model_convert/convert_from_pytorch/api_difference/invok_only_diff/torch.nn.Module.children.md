## [ 仅 API 调用方式不一致 ]torch.nn.Module.children

### [torch.nn.Module.children](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.children)

```python
torch.nn.Module.children()
```

### [paddle.nn.Layer.children](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/Layer/children_cn.html#paddle/nn/Layer/children_cn#cn-api-paddle-nn-Layer-children)

```python
paddle.nn.Layer.children()
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
net = torch.nn.Sequential(l, l)
result = torch.Tensor([0, 0])
for i, j in enumerate(net.children()):
    result = j(result)
# Paddle 写法
net = paddle.nn.Sequential(l, l)
result = paddle.Tensor([0, 0])
for i, j in enumerate(net.children()):
    result = j(result)
```
