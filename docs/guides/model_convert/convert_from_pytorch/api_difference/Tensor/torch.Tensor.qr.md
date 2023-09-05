## [ 仅参数名不一致 ] torch.Tensor.qr

### [torch.Tensor.qr](https://pytorch.org/docs/stable/generated/torch.Tensor.qr.html?highlight=torch+tensor+qr#torch.Tensor.qr)

```python
torch.Tensor.qr(some=True)
```

### [paddle.Tensor.qr](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/linalg/qr_cn.html#qr)

```python
paddle.Tensor.qr(mode='reduced')
```

两者功能一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                   |
|---------|--------------|----------------------|
| some    | mode         | 表示 QR 分解的行为。 需要进行转写。 |


### 转写示例
### some：控制 QR 分解的行为
```python
# 当进行完整的 QR 分解时
# Pytorch 写法
q, r = x.qr(some=False)

# Paddle 写法
q, r = x.qr(mode='complete')


#当进行减少的 QR 分解时
# Pytorch 写法
q, r = x.qr(some=True)

# Paddle 写法
q, r = x.qr(mode='reduced')
```
