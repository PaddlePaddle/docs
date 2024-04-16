## [ 仅参数名不一致 ]torch.istft
### [torch.istft](https://pytorch.org/docs/stable/generated/torch.istft.html?highlight=istft#torch.istft)

```python
torch.istft(input,
            n_fft,
            hop_length=None,
            win_length=None,
            window=None,
            center=True,
            normalized=False,
            onesided=None,
            length=None,
            return_complex=False)
```

### [paddle.signal.istft](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/signal/istft_cn.html#istft)

```python
paddle.signal.istft(x,
                    n_fft,
                    hop_length=None,
                    win_length=None,
                    window=None,
                    center=True,
                    normalized=False,
                    onesided=True,
                    length=None,
                    return_complex=False,
                    name=None)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 表示输入的 Tensor ，仅参数名不一致。                   |
| n_fft         | n_fft            | 表示离散傅里叶变换的样本点个数。                   |
| hop_length         | hop_length            | 表示相邻两帧偏移的样本点个数。                   |
| win_length         | win_length            | 表示信号窗的长度。                   |
| window         | window            | 表示长度为 win_length 的 Tensor 。                   |
| center         | center            | 表示是否将输入信号进行补长。                   |
| normalized         | normalized            | 表示是否将傅里叶变换的结果乘以值为 1/sqrt(n) 的缩放系数。                   |
| onesided         | onesided            | 表示是否返回一个实信号。                   |
| length         | length            | 表示输出信号的长度。                   |
| return_complex         | return_complex            | 表示输出的重构信号是否为复信号。                   |
