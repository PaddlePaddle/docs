## [ torch 参数更多 ]torch.nanmean

### [torch.nanmean](https://pytorch.org/docs/1.13/generated/torch.nanmean.html?highlight=nanmean#torch.nanmean)

```python
torch.nanmean(input,
              dim=None,
              keepdim=False,
              dtype=None,
              out=None)
```

### [paddle.nanmean](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nanmean_cn.html)

```python
paddle.nanmean(x,
               axis=None,
               keepdim=False,
               name=None)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input          | x         | 表示输入的 Tensor，仅参数名不一致。                                     |
| dim        | axis      | 表示进行运算的轴，可选项，仅参数名不一致。                |
| keepdim   | keepdim   | 表示是否保留计算后的维度，可选项。                    |
| dtype | - | 指定输出数据类型，可选项，Pytorch 默认值为 None，Paddle 无此参数，需要转写。 |
| out       | -        | 表示输出的 Tensor,可选项，Paddle 无此参数，需要进行转写。 |

### 转写示例

#### dytpe：指定数据类型

```python
# Pytorch 写法
torch.nanmean(x, dim=-1, dtype=torch.float32,out=y)

# Paddle 写法
paddle.assign(paddle.nanmean(x.astype('float32'),dim=-1),y)
```

#### out：指定输出

```python
# Pytorch 写法
torch.nanmean(t, dim=1，out=y)

# Paddle 写法
paddle.assign(paddle.nanmean(t, dim=1), y)
```
