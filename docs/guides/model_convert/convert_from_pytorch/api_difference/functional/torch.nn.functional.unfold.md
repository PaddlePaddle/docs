
## [ 参数名不一致 ]torch.nn.functional.unfold
### [torch.nn.functional.unfold](https://pytorch.org/docs/stable/generated/torch.nn.functional.unfold.html#torch.nn.functional.unfold)

```python
torch.nn.functional.unfold(input,
                           kernel_size,
                           dilation=1,
                           padding=0,
                           stride=1)
```

### [paddle.nn.functional.unfold](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/functional/unfold_cn.html#unfold)

```python
paddle.nn.functional.unfold(x,
                            kernel_size=[3, 3],
                            strides=1,
                            addings=1,
                            dilation=1,
                            name=None)
```
其中 Paddle 与 Pytorch 前四个参数所支持的参数类型不一致，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input     | x            | 输入 Tensor  |
| kernel_size   | kernel_sizes | 卷积核大小， PyTorch 支持 int、tuple(int)或者 list(int)，Paddle 仅支持 int 或者 list(int)   |
| dilation      | dilations    | 卷积膨胀，PyTorch 支持 int、tuple(int)或者 list(int)，Paddle 仅支持 int 或者 list(int) |
| padding       | paddings     | 每个维度的扩展，PyTorch 支持 int、tuple(int)或者 list(int)，Paddle 仅支持 int 或者 list(int) |
| stride        | strides      | 步长大小，PyTorch 支持 int、tuple(int)或者 list(int)，Paddle 仅支持 int 或者 list(int)|

### 转写示例
#### kernel_size: 卷积核大小
``` python
# PyTorch 写法
unfold = nn.functional.unfold(input,kernel_size=(2, 3))

# Paddle 写法
unfold = nn.functional.unfold(input,kernel_size=[2, 3])
```
