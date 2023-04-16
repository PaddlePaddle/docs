## [ 仅 paddle 参数更多 ] torch.fft.fftshift

### [torch.fft.fftshift](https://pytorch.org/docs/1.13/generated/torch.fft.fftshift.html#torch.fft.fftshift)

```python
torch.fft.fftshift(input, dim=None)
```

### [paddle.fft.fftshift](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/fft/fftshift_cn.html)

```python
paddle.fft.fftshift(x, axes=None, name=None)
```

Paddle 相比 PyTorch 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input           | x           | 输入 Tensor，仅参数名不一致。               |
| dim           | axes           | 进行移动的轴，仅参数名不一致。               |
| -           | name           | 网络前缀标识，PyTorch 无此参数，Paddle 保持默认即可。               |
