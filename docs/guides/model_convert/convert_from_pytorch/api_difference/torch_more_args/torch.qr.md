## [ torch 参数更多 ]torch.qr
### [torch.qr](https://docs.pytorch.org/docs/stable/generated/torch.qr.html#torch.qr)
```python
torch.qr(input, some=True, *, out=None)
```

### [paddle.qr](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/qr_cn.html)
```python
paddle.qr(input, some=True, *, out=None)
```

两者功能一致，参数映射如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input          | input        | 表示输入 Tensor。                                      |
| some           | some         | 表示 QR 分解的行为。                                   |
| out            | out          | 表示输出的 Tensor 元组。                               |

### 转写示例
#### 功能一致
```python
# PyTorch 写法
q, r = torch.qr(x, some=True)

# Paddle 写法
q, r = paddle.qr(x, some=True)
```
