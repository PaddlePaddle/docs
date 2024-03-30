## [torch 参数更多 ]torch.stft

### [torch.stft](https://pytorch.org/docs/stable/generated/torch.stft.html?highlight=stft#torch.stft)

```python
torch.stft(input,
           n_fft,
           hop_length=None,
           win_length=None,
           window=None,
           center=True,
           pad_mode='reflect',
           normalized=False,
           onesided=None,
           return_complex=None)
```

### [paddle.signal.stft](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/signal/stft_cn.html#paddle.signal.stft)

```python
paddle.signal.stft(x,
                   n_fft,
                   hop_length=None,
                   win_length=None,
                   window=None,
                   center=True,
                   pad_mode='reflect',
                   normalized=False,
                   onesided=True,
                   name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 输入的数据，是维度为 1D 或者 2D 的 Tensor， 数据类型可为复数（复信号），仅参数名不一致。   |
| n_fft         | n_fft        | 离散傅里叶变换的样本点个数。                                               |
| hop_length    | hop_length   | 对输入分帧时，相邻两帧偏移的样本点个数。默认为 None（为 n_ff/4）。          |
| win_length    | win_length   | 信号窗的长度。默认为 None（为 n_fft）。      |
| window        | window       | 维度为 1D 长度为 win_length 的 Tensor，数据类型可为复数。默认为 None。      |
| center        | center       | 选择是否将输入信号进行补长。默认为 True。                                   |
| pad_mode      | pad_mode     | 当 center 为 True 时，确定 padding 的模式。                              |
| normalized    | normalized   | 是否将傅里叶变换的结果乘以值为 1/sqrt(n) 的缩放系数。                       |
| onesided      | onesided     | 当输入为实信号时，选择是否只返回傅里叶变换结果的一半的频点值，如果输入的信号或者窗函数的 数据类型是复数，则此时不能设置为 True。默认为 True。      |
| return_complex| -            | 表示当输入为复数时，是否以复数形式返回，还是将实部与虚部分开以实数形式返回。Paddle 目前只支持返回复数，分开返回实部与虚部的情况，需要使用 as_real 进行转写。                |


### 转写示例
#### return_complex：是否返回复数
```python
# PyTorch 写法
y = torch.stft(input, n_fft=512, return_complex=False)

# Paddle 写法
y = paddle.as_real(paddle.stft(input, n_fft=512))
```
