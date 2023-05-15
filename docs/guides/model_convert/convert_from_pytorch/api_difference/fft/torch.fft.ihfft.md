## [torch 参数更多] torch.fft.ihfft

### [torch.fft.ihfft](https://pytorch.org/docs/1.13/generated/torch.fft.ihfft.html?highlight=ihfft#torch.fft.ihfft)

```python
torch.fft.ihfft(input,
                n=None,
                dim=- 1,
                norm=None,
                *,
                out=None)
```

### [paddle.fft.ihfft](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/fft/ihfft_cn.html)

```python
paddle.fft.ihfft(x,
                n=None,
                axis=- 1,
                norm='backward',
                name=None)
```

其中，PyTorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            |输入 Tensor，仅参数名不一致。                            |
| n             | n            |傅里叶变换点数。                                   |
| dim           | axis         |傅里叶变换的轴，如果没有指定，默认使用最后一维，仅参数名不一致。|
| norm           |norm          |指定傅里叶变换的缩放模式，缩放系数由变换的方向和模式同时决定。|
| out            | -            |输出的 Tensor,Paddle 无此参数，需要进行转写。              |

### 转写示例
#### out：指定输出
```python
# Pytorch 写法
torch.fft.ihfft(x, out=y)

# Paddle 写法
paddle.assign(paddle.fft.ihfft(x),y)
```
