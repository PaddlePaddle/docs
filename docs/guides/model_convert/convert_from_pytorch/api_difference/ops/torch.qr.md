## [ torch 参数更多 ] torch.qr

### [torch.qr](https://pytorch.org/docs/1.13/generated/torch.qr.html#torch.qr)

```python
torch.qr(input, some=True, *, out=None)
```

### [paddle.linalg.qr](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/linalg/qr_cn.html#qr)

```python
paddle.linalg.qr(x, mode='reduced', name=None)
```

其中 PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input          | x            | 表示输入 Tensor，仅参数名不一致。                           |
| some          | mode            | 表示 QR 分解的行为。 需进行转写。                        |
| out          | -            | 表示输出的 Tensor 元组。 Paddle 无此参数，需要进行转写。                           |

### 转写示例
### some：控制 QR 分解的行为
```python
# Pytorch 写法
q, r = torch.qr(x, some=False)

# Paddle 写法
q, r = paddle.linalg.qr(x, mode='complete')
```

#### out：指定输出
```python
# Pytorch 写法
torch.qr(x, out = (q, r) )

# Paddle 写法
q, r = paddle.linalg.qr(x)
```
