## [torch 参数更多 ]torch.special.log1p
### [torch.special.log1p](https://pytorch.org/docs/stable/special.html#torch.special.log1p)

```python
torch.special.log1p(input,
            *,
            out=None)
```

### [paddle.log1p](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/log1p_cn.html#log1p)

```python
paddle.log1p(x,
             name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| <font color='red'> input </font> | <font color='red'> x </font> | 表示输入的 Tensor ，仅参数名不一致。  |
| <font color='red'> out </font> | -  | 表示输出的 Tensor ， Paddle 无此参数，需要转写。    |


### 转写示例
#### out：指定输出
```python
# PyTorch 写法
torch.special.log1p(input, out=y)

# Paddle 写法
paddle.assign(paddle.log1p(input), y)
```
