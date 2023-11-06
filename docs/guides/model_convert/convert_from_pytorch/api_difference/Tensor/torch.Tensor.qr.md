## [ 参数不一致 ] torch.Tensor.qr

### [torch.Tensor.qr](https://pytorch.org/docs/stable/generated/torch.Tensor.qr.html?highlight=torch+tensor+qr#torch.Tensor.qr)

```python
torch.Tensor.qr(some=True)
```

### [paddle.Tensor.qr](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/linalg/qr_cn.html#qr)

```python
paddle.Tensor.qr(mode='reduced')
```

其中，PyTorch 的 `some` 和 PaddlePaddle 的 `mode` 参数所支持的数据类型不一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                           |
|---------|--------------|--------------------------------------------------------------|
| some    | mode         | 表示 QR 分解的行为。PyTorch 支持布尔类型的输入，PaddlePaddle 支持字符串类型的输入。 需要转写。 |


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
