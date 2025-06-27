## [ 仅参数名不一致 ]torch.Tensor.igamma

### [torch.Tensor.igamma](https://pytorch.org/docs/stable/generated/torch.Tensor.igamma.html#torch.Tensor.igamma)

```python
torch.Tensor.igamma(other)
```

### [paddle.Tensor.gammainc](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/gammainc_cn.html#gammainc)

```python
paddle.Tensor.gammainc(y, name=None)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                        |
| ------- | ------------ | --------------------------- |
| other   | y            | 正参数 Tensor，仅参数名不同 |

### 转写示例

```python
# PyTorch 写法
out = x.igamma(y)

# Paddle 写法
out = x.gammainc(y)
```
