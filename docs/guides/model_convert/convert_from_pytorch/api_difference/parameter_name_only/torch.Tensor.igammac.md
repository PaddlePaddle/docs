## [ 仅参数名不一致 ]torch.Tensor.igammac

### [torch.Tensor.igammac](https://pytorch.org/docs/stable/generated/torch.Tensor.igammac.html#torch.Tensor.igammac)

```python
torch.Tensor.igammac(other)
```

### [paddle.Tensor.gammaincc](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/gammaincc_cn.html#gammaincc)

```python
paddle.Tensor.gammaincc(y, name=None)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                        |
| ------- | ------------ | --------------------------- |
| other   | y            | 正参数 Tensor，仅参数名不同 |

### 转写示例

```python
# PyTorch 写法
out = x.igammac(y)

# Paddle 写法
out = x.gammaincc(y)
```
