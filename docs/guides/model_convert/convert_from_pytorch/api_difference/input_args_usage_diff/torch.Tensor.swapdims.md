## [ 输入参数用法不一致 ] torch.Tensor.swapdims

### [torch.Tensor.swapdims](https://pytorch.org/docs/stable/generated/torch.Tensor.swapdims.html#torch.Tensor.swapdims)

```python
torch.Tensor.swapdims(dim0, dim1)
```

### [paddle.Tensor.transpose](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#transpose-perm-name-none)

```python
paddle.Tensor.transpose(perm,
                 name=None)
```

其中 PyTorch 的 `dim0、dim1` 与 Paddle 用法不一致，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| dim0          | -            | PyTorch 转置的第一个维度，Paddle 无此参数，需要转写。                  |
| dim1          | -            | PyTorch 转置的第二个维度，Paddle 无此参数，需要转写。                   |
| -             | perm         | Paddle 无此参数。 Paddle 可通过 perm 参数，等价的实现 torch 的 dim0、dim1 的功能。|


### 转写示例

#### dim0、dim1 参数： 转置的维度设置
``` python
# PyTorch 写法:
x.swapaxes(dim0=0, dim1=1)

# Paddle 写法:
x.transpose(perm=[1, 0, 2])

# 注：x 为 3D Tensor
```
