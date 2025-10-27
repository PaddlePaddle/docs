## [ 仅 API 调用方式不一致 ]torch.nn.Module.named_children

### [torch.nn.Module.named_children](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.named_children)

```python
torch.nn.Module.named_children()
```

### [paddle.nn.Layer.named_children](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/Layer/named_children_cn.html#paddle/nn/Layer/named_children_cn#cn-api-paddle-nn-Layer-named_children)

```python
paddle.nn.Layer.named_children()
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
model = torch.nn.Sequential(OrderedDict([("wfs", l), ("wfs1", l1)]))
result = torch.Tensor([0, 0])
for name, module in model.named_children():
    result = module(result)
# Paddle 写法
model = paddle.nn.Sequential(OrderedDict([("wfs", l), ("wfs1", l1)]))
result = paddle.Tensor([0, 0])
for name, module in model.named_children():
    result = module(result)
```
