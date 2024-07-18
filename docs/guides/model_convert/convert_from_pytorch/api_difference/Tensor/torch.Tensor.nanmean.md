## [ torch 参数更多 ]torch.Tensor.nanmean

### [torch.Tensor.nanmean](https://pytorch.org/docs/stable/generated/torch.Tensor.nanmean.html?highlight=nanmean#torch.Tensor.nanmean)

```python
torch.Tensor.nanmean(dim=None,
                     keepdim=False,
                     dtype=None,
                     out=None)
```

### [paddle.Tensor.nanmean](暂无对应文档)

```python
paddle.Tensor.nanmean(axis=None,
                      keepdim=False,
                      name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| dim        | axis      | 表示进行运算的轴，可选项，仅参数名不一致。                |
| keepdim   | keepdim   | 表示是否保留计算后的维度，可选项。                    |
| dtype | - | 指定输出数据类型，可选项，PyTorch 默认值为 None，Paddle 无此参数，需要转写。 |
| out       | -        | 表示输出的 Tensor,可选项，Paddle 无此参数，需要转写。 |

### 转写示例

#### dytpe：指定数据类型

```python
# PyTorch 写法
x.nanmean(dim=-1, dtype=torch.float32,out=y)

# Paddle 写法
x.astype('float32')
paddle.assign(x.nanmean(dim=-1),y)
```

#### out：指定输出

```python
# PyTorch 写法
x.nanmean(dim=1，out=y)

# Paddle 写法
paddle.assign(x.nanmean(dim=1), y)
```
