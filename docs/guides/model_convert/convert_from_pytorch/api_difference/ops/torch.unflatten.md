## [ 参数不⼀致 ] torch.unflatten

### [torch.unflatten](https://pytorch.org/docs/stable/generated/torch.unflatten.html#torch.unflatten)

```python
torch.unflatten(input, dim, sizes)
```

### [paddle.reshape](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/reshape_cn.html#reshape)

```python
paddle.reshape(x, shape, name=None)
```

其中 Pytorch 的 `dim、sizes` 与 Paddle 用法不一致，具体如下：

### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| <font color='red'>input</font>         | <font color='red'>x</font>            | 输入 Tensor。                                       |
| <font color='red'>dim</font>          | -            | Pytorch 需要变换的维度，Paddle 无此参数，需要转写。 Paddle 可修改 shape 实现。                   |
| <font color='red'>sizes</font>          | -            | Pytorch 维度变换的新形状，Paddle 无此参数，需要转写。Paddle 可修改 shape 实现。                    |
| -             | <font color='red'>shape</font>         | Paddle 可通过 shape 参数，等价的实现 torch 的 dim、sizes 的功能。|


### 转写示例

#### dim、sizes 参数： 转置的维度设置
``` python
# PyTorch 写法:
torch.unflatten(input, -1, (2, 2))

# Paddle 写法:
paddle.reshape(input, shape=[3, 2, 2])

# 注：input 的形状为[3, 4]
```
