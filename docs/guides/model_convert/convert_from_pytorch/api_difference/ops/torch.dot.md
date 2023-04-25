## [torch 参数更多 ]torch.dot

### [torch.dot](https://pytorch.org/docs/stable/generated/torch.dot.html?highlight=dot#torch.dot)

```python
torch.dot(input, other, *, out=None)
```

### [paddle.abs](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/dot_cn.html#dot)

```python
paddle.dot(x, y, name=None)
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| <font color='red'> input </font>         | <font color='red'> x </font>            | 输入的 Tensor ，仅参数名不同。                                     |
| <font color='red'> other </font>         | <font color='red'> y </font>            | 输入的 Tensor ，仅参数名不同。                                     |
| <font color='red'> out </font>           | -                                       | 表示输出的 Tensor，PaddlePaddle 无此参数，需要进行转写。              |


### 转写示例

#### out：指定输出
```python
# Pytorch 写法
torch.dot(torch.tensor([2, 3]), torch.tensor([2, 1]), out=y)

# Paddle 写法
y = paddle.dot(paddle.to_tensor([2, 3]), paddle.to_tensor([2, 1]))
```
