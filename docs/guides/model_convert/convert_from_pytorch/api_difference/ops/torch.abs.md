## [torch 参数更多 ]torch.abs

### [torch.abs](https://pytorch.org/docs/stable/generated/torch.abs.html?highlight=abs#torch.abs)

```python
torch.abs(input,
          *,
          out=None)
```

### [paddle.abs](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/abs_cn.html#abs)

```python
paddle.abs(x,
           name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| <font color='red'> input </font>         | <font color='red'> x </font>            | 输入的 Tensor ，仅参数名不一致。                                     |
| <font color='red'> out </font>           | -            | 表示输出的 Tensor，Paddle 无此参数，需要转写。              |


### 转写示例

#### out：指定输出
```python
# PyTorch 写法
torch.abs([-3, -5], out=y)

# Paddle 写法
paddle.assign(paddle.abs([-3, -5]), y)
```
