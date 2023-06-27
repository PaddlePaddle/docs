## [ 仅参数名不一致 ] torch.fft.ifftshift

### [torch.fft.ifftshift](https://pytorch.org/docs/1.13/generated/torch.fft.ifftshift.html#torch.fft.ifftshift)

```python
torch.fft.ifftshift(input, dim=None)
```

### [paddle.fft.ifftshift](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/fft/ifftshift_cn.html)

```python
paddle.fft.ifftshift(x, axes=None, name=None)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input           | x           | 输入 Tensor，仅参数名不一致。               |
| dim           | axes           | 进行移动的轴，仅参数名不一致。               |

### 转写示例
#### other
```python
# PyTorch 写法
torch.fft.ifftshift(torch.tensor([ 0, 1]), dim=0)

# Paddle 写法
torch.fft.ifftshift(paddle.to_tensor([ 0, 1]), axes=0)
```
