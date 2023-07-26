## [ torch 参数更多 ]torch.fft.irfft

### [torch.fft.irfft](https://pytorch.org/docs/stable/generated/torch.fft.irfft.html#torch-fft-irfft)

```python
torch.fft.irfft(input, n=None, dim=- 1, norm='backward', *, out=None)
```

### [paddle.fft.irfft](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/fft/irfft_cn.html#irfft)

```python
paddle.fft.irfft(x, n=None, axis=- 1, norm='backward', name=None)
```

其中，Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch                             | PaddlePaddle | 备注                                                                    |
| ----------------------------------- | ------------ | ----------------------------------------------------------------------- |
| input     | x           | 表示输入的 Tensor ，仅参数名不一致。                         |
| n     | n           | 表示在傅里叶变换轴的长度 。                         |
| dim       | axis        | 表示进行运算的轴，仅参数名不一致。                           |
| norm     | norm           | 表示傅里叶变换的缩放模式。                         |
| out           | -      | 表示输出的 Tensor ， Paddle 无此参数，需要进行转写。         |

###  转写示例
#### out：指定输出
```python
# Pytorch 写法
torch.fft.irfft(x, s, dim, norm, out=y)

# Paddle 写法
paddle.assign(paddle.fft.irfft(x, s, dim, norm) , y)
```
