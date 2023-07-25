## [ torch 参数更多 ] torch.Tensor.qr

### [torch.Tensor.qr](https://pytorch.org/docs/stable/generated/torch.linalg.qr.html?highlight=qr#torch.linalg.qr)

```python
torch.Tensor.qr(some=True)
```

### [paddle.Tensor.qr](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/linalg/qr_cn.html#qr)

```python
paddle.Tensor.qr(mode='reduced')
```

其中 PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注             |
|--------|-------------|----------------|
| mode   | some        | 表示 QR 分解的行为。 需进行转写。 |


### 转写示例

```python
# 当进行完整的 QR 分解时
# Pytorch 写法
x = torch.tensor([[1., 2., 3.],[4., 5., 6.,]])
q, r = x.qr(some=False)

# Paddle 写法
x = paddle.to_tensor([[1., 2., 3.],[4., 5., 6.,]])
q, r = x.qr(mode='complete')

#当进行减少的 QR 分解时
# Pytorch 写法
q, r = x.qr(some=True)

# Paddle 写法
q, r = x.qr(mode='reduced')
```
