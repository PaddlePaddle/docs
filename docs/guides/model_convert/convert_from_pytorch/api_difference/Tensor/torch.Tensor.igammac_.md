## [ 仅参数名不一致 ]torch.Tensor.igammac\_

### [torch.Tensor.igammac_](https://pytorch.org/docs/stable/generated/torch.Tensor.igammac_.html#torch.Tensor.igammac_)

```python
torch.Tensor.igammac_(other)
```

### [paddle.Tensor.gammaincc_](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/gammaincc__cn.html#gammaincc)

```python
paddle.Tensor.gammaincc_(y, name=None)
```

### 参数映射

| PyTorch | PaddlePaddle | 备注          |
| ------- | ------------ | ------------- |
| other   | y            | 正参数 Tensor |

### 转写示例

```python
# PyTorch 写法
out = x.igammac_(y)

# Paddle 写法
out = x.gammaincc_(y)
```
