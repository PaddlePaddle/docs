## [ 仅 API 调用方式不一致 ]torch.nn.Module.apply

### [torch.nn.Module.apply](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.apply)

```python
torch.nn.Module.apply(fn)
```

### [paddle.nn.Layer.apply](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/Layer/apply_cn.html#paddle/nn/Layer/apply_cn#cn-api-paddle-nn-Layer-apply)

```python
paddle.nn.Layer.apply(fn)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
def init_weights(m):
    pass
net = torch.nn.Sequential(
    torch.nn.Linear(2, 2, bias=False), torch.nn.Linear(2, 2, bias=False)
)
net.apply(init_weights)

# Paddle 写法
def init_weights(m):
    pass
net = paddle.nn.Sequential(
    paddle.nn.Linear(in_features=2, out_features=2, bias_attr=False),
    paddle.nn.Linear(in_features=2, out_features=2, bias_attr=False),
)
net.apply(init_weights)
```
