## [torch 参数更多 ]torch.multiply
### [torch.multiply](https://pytorch.org/docs/stable/generated/torch.multiply.html?highlight=multiply#torch.multiply)

```python
torch.multiply(input,
               other,
               *,
               out=None)
```

### [paddle.multiply](https://vpaddlepaddle.org.cn/documentation/docs/zh/api/paddle/multiply_cn.html#multiply)

```python
paddle.multiply(x,
                y,
                name=None)
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| <font color='red'> input </font> | <font color='red'> x </font> | 表示输入的 Tensor ，仅参数名不一致。  |
| <font color='red'> other </font> | <font color='red'> y </font> | 表示输入的 Tensor ，仅参数名不一致。  |
| <font color='red'> out </font> | -  | 表示输出的 Tensor ， Paddle 无此参数，需要进行转写。    |


### 转写示例
#### out：指定输出
```python
# Pytorch 写法
torch.multiply([[3, 5]], [[1], [2]], out=y)

# Paddle 写法
y = paddle.multiply([[3, 5]], [[1], [2]])
```
