## [torch 参数更多 ]torch.log2
### [torch.log2](https://pytorch.org/docs/stable/generated/torch.log2.html?highlight=log2#torch.log2)

```python
torch.log2(input,
           *,
           out=None)
```

### [paddle.log2](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/log2_cn.html#log2)

```python
paddle.log2(x,
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
torch.log2(input, out=y)

# Paddle 写法
paddle.assign(paddle.log2(input), y)
```
