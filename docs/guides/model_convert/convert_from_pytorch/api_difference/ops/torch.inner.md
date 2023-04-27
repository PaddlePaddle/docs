## [torch 参数更多 ]torch.inner

### [torch.inner](https://pytorch.org/docs/1.13/generated/torch.inner.html?highlight=inner#torch.inner)

```python
torch.inner(input, other, *, out=None)
```

### [paddle.inner](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/inner_cn.html)

```python
paddle.inner(x, y, name=None)
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
torch.inner(torch.arange(1., 7.).reshape((2, 3)), torch.arange(1., 10.).reshape((3, 3)), out=y)

# Paddle 写法
paddle.assign(paddle.inner(paddle.arange(1, 7).reshape((2, 3)).astype('float32'), paddle.arange(1, 10).reshape((3, 3)).astype('float32')), y)
```
