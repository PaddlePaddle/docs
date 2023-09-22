## [torch 参数更多 ]torch.inner

### [torch.inner](https://pytorch.org/docs/stable/generated/torch.inner.html?highlight=inner#torch.inner)

```python
torch.inner(input, other, *, out=None)
```

### [paddle.inner](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/inner_cn.html)

```python
paddle.inner(x, y, name=None)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| <font color='red'> input </font>         | <font color='red'> x </font>            | 输入的 Tensor ，仅参数名不一致。                                     |
| <font color='red'> other </font>         | <font color='red'> y </font>            | 输入的 Tensor ，仅参数名不一致。                                     |
| <font color='red'> out </font>           | -                                       | 表示输出的 Tensor，Paddle 无此参数，需要转写。              |


### 转写示例

#### out：指定输出
```python
# Pytorch 写法
torch.inner([[1., 2. , 3.], [4. ,5. ,6.]], [[1., 2. , 3.], [4. ,5. ,6.], [7., 8., 9.]], out=y)

# Paddle 写法
paddle.assign(paddle.inner([[1., 2. , 3.], [4. ,5. ,6.]], [[1., 2. , 3.], [4. ,5. ,6.], [7., 8., 9.]]), y)
```
