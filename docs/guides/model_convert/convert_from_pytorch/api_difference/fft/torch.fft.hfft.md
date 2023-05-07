## [torch 参数更多] torch.fft.hfft

### [torch.fft.hfft](https://pytorch.org/docs/stable/generated/torch.fft.hfft.html?highlight=hfft#torch.fft.hfft)

```python
torch.fft.hfft(input,
                n=None,
                dim=- 1,
                norm=None,
                *,
                out=None)
```

### [paddle.fft.hfft](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/fft/hfft_cn.html)

```python
paddle.fft.hfft(x,
                n=None,
                axis=- 1,
                norm='backward',
                name=None)
```

两者功能一致但参数不一致，部分参数名不同，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 输入 Tensor，仅参数名不一致。                            |
| n             | n            | 输出 Tensor 在傅里叶变换轴的长度。 参数名相同。    |
| dim           | axis         | 傅里叶变换的轴。如果没有指定，默认是使用最后一维。仅参数名不一致|
| norm           |norm          |指定傅里叶变换的缩放模式，缩放系数由变换的方向和模式同时决定。参数名相同。|
| out            | -            |输出的 Tensor,Paddle 无此参数，需要进行转写。              |

### 转写示例
#### out：指定输出
```python
# Pytorch 写法
torch.fft.hfft(tensor([ 0.5000-0.0000j, -0.1250-0.1720j, -0.1250-0.0406j, -0.1250+0.0406j,
-0.1250+0.1720j])[:3], n=5, out=y)

# Paddle 写法
paddle.assign(paddle.fft.hfft(paddle.to_tensor([1, -1j, -1])),y)
```
