## torch.istft
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

### [paddle.signal.istft](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/signal/istft_cn.html#istft)

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

两者功能一致且参数用法一致，仅参数名不同，具体如下：
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 输入的 Tensor。                   |
