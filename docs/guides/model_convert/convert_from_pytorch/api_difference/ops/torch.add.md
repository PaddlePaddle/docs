## [torch 参数更多 ]torch.add

### [torch.add](https://pytorch.org/docs/stable/generated/torch.add.html?highlight=torch+add#torch.add)

```python
torch.add(input,
          other,
          *,
          alpha=1,
          out=None)
```

### [paddle.add](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/add_cn.html#add)

```python
paddle.add(x,
           y,
           name=None)
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| <font color='red'> input </font>         | <font color='red'> x </font>           | 输入的 Tensor ,仅参数名不同。                                     |
| <font color='red'> other </font>         | <font color='red'> y </font>            | 输入的 Tensor ,仅参数名不同。                                     |
| <font color='red'> alpha </font>        | -            | other 的乘数，PaddlePaddle 无此参数，需要进行转写。                   |
| <font color='red'> out </font>           | -            | 表示输出的 Tensor，PaddlePaddle 无此参数，需要进行转写。               |


### 转写示例
#### alpha：other 的乘数
```python
# Pytorch 写法
torch.add([3, 5], [2, 3], alpha=2)

# Paddle 写法
paddle.add([3, 5], 2 * [2, 3])
```

#### out：指定输出
```python
# Pytorch 写法
torch.add([3, 5], [2, 3], out=y)

# Paddle 写法
y = paddle.add([3, 5], [2, 3])
```
