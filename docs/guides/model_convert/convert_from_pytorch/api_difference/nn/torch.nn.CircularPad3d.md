
## [ API 可直接映射 ]torch.nn.CircularPad3d

### [torch.nn.CircularPad3d](https://pytorch.org/docs/stable/generated/torch.nn.CircularPad3d.html#circularpad3d)

```python
torch.nn.CircularPad3d(padding)
```

### [paddle.nn.Pad3D](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/Pad3D_cn.html#pad3d)

```python
paddle.nn.Pad3D(padding, mode='constant', value=0.0, data_format='NCDHW', name=None)
```

### 转写示例

```python
# PyTorch 写法
module = torch.nn.CircularPad3d(3)
module(input)

# Paddle 写法
module = paddle.nn.Pad3D(3, mode="circular")
module(input)
```
