## [ 仅参数名不一致 ]torch.Tensor.igamma_

### [torch.Tensor.igamma_](https://pytorch.org/docs/stable/generated/torch.Tensor.igamma_.html#torch.Tensor.igamma_)

```python
torch.Tensor.igamma_(other)
```

### [paddle.Tensor.gammainc_](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/gammainc__cn.html#gammainc)

```python
paddle.Tensor.gammainc_(y, name=None)
```

### 参数映射

| PyTorch | PaddlePaddle | 备注          |
| ------- | ------------ | ------------- |
| other   | y            | 正参数 Tensor |

### 转写示例

```python
# PyTorch 写法
out = x.igamma_(y)

# Paddle 写法
out = x.gammainc_(y)
```
