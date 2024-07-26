## [ torch 参数更多 ] torch.where

### [torch.where](https://pytorch.org/docs/stable/generated/torch.where.html)

```python
torch.where(condition, input, other, *, out=None)
```

### [paddle.where](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/where_cn.html)

```python
paddle.where(condition, x, y, name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| condition     | condition    | 判断条件。                             |
| input         | x            | 当 condition 为 true 时，选择 input 元素，仅参数名不一致。 |
| other         | y            | 当 condition 为 false 时，选择 other 中元素，仅参数名不一致。 |
| out           | -            | 表示输出的 Tensor，Paddle 无此参数，需要转写。 |

### 转写示例

#### out：指定输出

```python
# PyTorch 写法
torch.where(x > 0, x, y, out=z)

# Paddle 写法
paddle.assign(paddle.where(x > 0, x, y), z)
```
