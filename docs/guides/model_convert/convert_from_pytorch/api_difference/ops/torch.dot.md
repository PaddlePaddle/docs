## [torch 参数更多 ]torch.dot

### [torch.dot](https://pytorch.org/docs/stable/generated/torch.dot.html?highlight=dot#torch.dot)

```python
torch.dot(input, other, *, out=None)
```

### [paddle.dot](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/dot_cn.html#dot)

```python
paddle.dot(x, y, name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| <font color='red'> input </font>         | <font color='red'> x </font>            | 输入的 Tensor ，仅参数名不一致。                                     |
| <font color='red'> other </font>         | <font color='red'> y </font>            | 输入的 Tensor ，仅参数名不一致。                                     |
| <font color='red'> out </font>           | -                                       | 表示输出的 Tensor，Paddle 无此参数，需要转写。              |


### 转写示例

#### out：指定输出
```python
# PyTorch 写法
torch.dot(torch.tensor([2, 3]), torch.tensor([2, 1]), out=y)

# Paddle 写法
paddle.assign(paddle.dot(paddle.to_tensor([2, 3]), paddle.to_tensor([2, 1])), y)
```
