## [ 仅 API 调用方式不一致 ]torch.nn.MaxUnpool3d

### [torch.nn.MaxUnpool3d](https://pytorch.org/docs/stable/generated/torch.nn.MaxUnpool3d.html#torch.nn.MaxUnpool3d)

```python
torch.nn.MaxUnpool3d(kernel_size, stride=None, padding=0)
```

### [paddle.nn.MaxUnPool3D](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/MaxUnPool3D_cn.html#paddle/nn/MaxUnPool3D_cn#cn-api-paddle-nn-MaxUnPool3D)

```python
paddle.nn.MaxUnPool3D(kernel_size, stride=None, padding=0, data_format='NCDHW', output_size=None, name=None)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
unpool = torch.nn.MaxUnpool3d(3, stride=2)

# Paddle 写法
unpool = paddle.nn.MaxUnPool3D(3, stride=2)
```
