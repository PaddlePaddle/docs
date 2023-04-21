## [ 参数用法不⼀致 ] torch.swapdims

### [torch.swapdims](https://pytorch.org/docs/master/generated/torch.swapdims.html#torch.swapdims)

```python
torch.swapdims(input,
                dim0,
                dim1)
```

### [paddle.transpose](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/transpose_cn.html#transpose)

```python
paddle.transpose(x,
                 perm,
                 name=None)
```

其中 Pytorch 的 `dim0、dim1` 与 Paddle 用法不一致，具体如下：

### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| <font color='red'>input</font>         | <font color='red'>x</font>            | 输入 Tensor。                                       |
| <font color='red'>dim0</font>          | -            | Pytorch 转置的第一个维度，Paddle 无此参数，需要进行转写。                    |
| <font color='red'>dim1</font>          | -            | Pytorch 转置的第二个维度，Paddle 无此参数，需要进行转写。                    |
| -             | <font color='red'>perm</font>         | Paddle 可通过 perm 参数，等价的实现 torch 的 dim0、dim1 的功能。|


### 转写示例

#### dim0、dim1 参数： 转置的维度设置
``` python
# PyTorch 写法:
torch.swapdims(x, dim0=0, dim1=1)

# Paddle 写法:
paddle.transpose(x, perm=[1, 0, 2])

# 注：x 为 3D Tensor
```
