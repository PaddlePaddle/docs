## [ torch 参数更多 ] torch.fft.ifftn

### [torch.fft.ifftn](https://pytorch.org/docs/stable/generated/torch.fft.ifftn.html?highlight=ifftn#torch.fft.ifftn)

```python
torch.fft.ifftn(input,
                s=None,
                dim=None,
                norm=None,
                *,
                out=None)
```

### [paddle.fft.ifftn](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/fft/ifftn_cn.html)

```python
paddle.fft.ifftn(x,
                s=None,
                axes=None,
                norm='backward',
                name=None)
```

其中，PyTorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            |输入 Tensor，仅参数名不一致。                            |
| s             | s            |输出 Tensor 在每一个傅里叶变换轴上的长度。               |
| dim           | axes         |计算快速傅里叶变换的轴。仅参数名不一致。                  |
| norm           |norm          |傅里叶变换的缩放模式，缩放系数由变换的方向和缩放模式同时决定，完全一致。|
| out            | -            |输出的 Tensor,Paddle 无此参数，需要进行转写。  |

### 转写示例
#### out：指定输出
```python
# Pytorch 写法
torch.fft.ifftn(x, out=y)

# Paddle 写法
paddle.assign(paddle.fft.ifftn(x), y)
```
