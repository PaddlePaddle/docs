## [torch 参数更多] torch.fft.irfftn

### [torch.fft.irfftn](https://pytorch.org/docs/stable/generated/torch.fft.irfftn.html?highlight=irfftn#torch.fft.irfftn)

```python
torch.fft.irfftn(input,
                s=None,
                dim=None,
                norm=None,
                *,
                out=None)
```

### [paddle.fft.irfftn](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/fft/irfftn_cn.html)

```python
paddle.fft.irfftn(x,
                s=None,
                axes=None,
                norm='backward',
                name=None)
```

两者功能一致但参数不一致，部分参数名不同，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 输入 Tensor，仅参数名不一致。                            |
| s             | s            | 输出 Tensor 在每一个傅里叶变换轴上的长度,参数名相同。          |
| dim           | axes         | 计算快速傅里叶变换的轴。仅参数名不一致。   |
| norm           |norm          |指定傅里叶变换的缩放模式，缩放系数由变换的方向和模式同时决定。参数名相同。|
| out            | -            |输出的 Tensor,Paddle 无此参数，需要进行转写。                     |
### 转写示例
#### out：指定输出
```python
# Pytorch 写法
torch.fft.rfftn(torch.rand(10, 9), out=y)

# Paddle 写法
paddle.assign(paddle.fft.irfftn(paddle.to_tensor([2.+2.j, 2.+2.j, 3.+3.j]).astype(paddle.complex128)),y)
```
