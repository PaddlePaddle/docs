## [ 仅参数名不一致 ] torch.fft.fftshift

### [torch.fft.fftshift](https://pytorch.org/docs/1.13/generated/torch.fft.fftshift.html#torch.fft.fftshift)

```python
torch.fft.fftshift(input, dim=None)
```

### [paddle.fft.fftshift](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/fft/fftshift_cn.html)

```python
paddle.fft.fftshift(x, axes=None, name=None)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input           | x           | 输入 Tensor，仅参数名不一致。               |
| dim           | axes           | 进行移动的轴，仅参数名不一致。               |
