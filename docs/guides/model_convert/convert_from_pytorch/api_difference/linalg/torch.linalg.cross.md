## [ torch 参数更多 ]torch.linalg.cross

### [torch.linalg.cross](https://pytorch.org/docs/stable/generated/torch.linalg.cross.html?highlight=torch+linalg+cross#torch.linalg.cross)

```python
torch.linalg.cross(input, other, *, dim=- 1, out=None)
```

### [paddle.cross](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/cross_cn.html)

```python
paddle.cross(x, y, axis=None, name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch                             | PaddlePaddle | 备注                                                                    |
| ----------------------------------- | ------------ | ----------------------------------------------------------------------- |
| input         | x      | 表示输入的 Tensor ，仅参数名不一致。                         |
| other         | y      | 表示输入的 Tensor ，仅参数名不一致。                         |
| dim       | axis        | 表示进行运算的维度，参数默认值不一致。PyTorch 默认为`-1`，Paddle 默认为 `None`。                           |
| out           | -      | 表示输出的 Tensor ， Paddle 无此参数，需要转写。         |

###  转写示例
#### out：指定输出
```python
# PyTorch 写法
torch.linalg.cross(input, other, dim=1, out=y)

# Paddle 写法
paddle.assign(paddle.cross(input, other, axis=1) , y)
```
