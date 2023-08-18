## [torch 参数更多 ]torch.add

### [torch.add](https://pytorch.org/docs/stable/generated/torch.add.html?highlight=add#torch.add)

```python
torch.add(input,
          other,
          *,
          alpha=1,
          out=None)
```

### [paddle.add](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/add_cn.html#add)

```python
paddle.add(x,
           y,
           name=None)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                     |
| ------- | ------------ | -------------------------------------------------------- |
| input   | x            | 表示输入的 Tensor ，仅参数名不一致。                     |
| other   | y            | 表示输入的 Tensor ，仅参数名不一致。                     |
| alpha   | -            | 表示 other 的乘数，Paddle 无此参数，需要转写。   |
| out     | -            | 表示输出的 Tensor，Paddle 无此参数，需要转写。 |


### 转写示例

#### alpha：other 的乘数

```python
# Pytorch 写法
torch.add(torch.tensor([3, 5]), torch.tensor([2, 3]), alpha=2)

# Paddle 写法
paddle.add(paddle.to_tensor([3, 5]), 2 * paddle.to_tensor( [2, 3]))
```

#### out：指定输出

```python
# Pytorch 写法
torch.add(torch.tensor([3, 5]), torch.tensor([2, 3]), out=y)

# Paddle 写法
paddle.assign(paddle.add(paddle.to_tensor([3, 5]), paddle.to_tensor([2, 3])), y)
```
