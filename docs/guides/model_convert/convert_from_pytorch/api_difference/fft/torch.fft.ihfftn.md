## [ torch 参数更多 ]torch.fft.ihfftn

### [torch.fft.ihfftn](https://pytorch.org/docs/stable/generated/torch.fft.ihfftn.html?highlight=torch+fft+ihfftn#torch.fft.ihfftn)

```python
torch.fft.ihfftn(input, s=None, dim=None, norm='backward', *, out=None)
```

### [paddle.fft.ihfftn](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/fft/ihfftn_cn.html)

```python
paddle.fft.ihfftn(x, s=None, axes=None, norm='backward', name=None)
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
torch.fft.ihfftn(x, s, dim, norm, out=y)

# Paddle 写法
paddle.assign(paddle.fft.ihfftn(x, s, dim, norm) , y)
```
