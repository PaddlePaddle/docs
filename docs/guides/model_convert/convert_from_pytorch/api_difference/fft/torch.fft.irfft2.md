## [ torch 参数更多 ]torch.fft.irfft2

### [torch.fft.irfft2](https://pytorch.org/docs/stable/generated/torch.fft.irfft2.html#torch-fft-irfft2)

```python
torch.fft.irfft2(input, s=None, dim=(- 2, - 1), norm='backward', *, out=None)
```

### [paddle.fft.irfft2](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/fft/irfft2_cn.html#irfft2)

```python
paddle.fft.irfft2(x, s=None, axes=(- 2, - 1), norm='backward', name=None)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch                             | PaddlePaddle | 备注                                                                    |
| ----------------------------------- | ------------ | ----------------------------------------------------------------------- |
| input     | x           | 表示输入的 Tensor ，仅参数名不一致。                         |
| s     | s           | 表示在傅里叶变换轴的长度 。                         |
| dim       | axes        | 表示进行运算的轴，仅参数名不一致。                           |
| norm     | norm           | 表示傅里叶变换的缩放模式。                         |
| out           | -      | 表示输出的 Tensor ， Paddle 无此参数，需要转写。         |

###  转写示例
#### out：指定输出
```python
# Pytorch 写法
torch.fft.irfft2(x, s, dim, norm, out=y)

# Paddle 写法
paddle.assign(paddle.fft.irfft2(x, s, dim, norm) , y)
```
