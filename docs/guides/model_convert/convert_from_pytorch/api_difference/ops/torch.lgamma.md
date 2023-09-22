## [torch 参数更多 ]torch.lgamma
### [torch.lgamma](https://pytorch.org/docs/stable/generated/torch.lgamma.html?highlight=lgamma#torch.lgamma)

```python
torch.lgamma(input,
              *,
              out=None)
```

### [paddle.lgamma](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/lgamma_cn.html#lgamma)

```python
paddle.lgamma(x
              , name=None)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数差异
|    PyTorch        | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| <font color='red'> input </font> | <font color='red'> x </font> | 表示输入的 Tensor ，仅参数名不一致。  |
| <font color='red'> out </font> | -  | 表示输出的 Tensor ， Paddle 无此参数，需要转写。    |



### 转写示例
#### out：指定输出
```python
# Pytorch 写法
torch.lgamma(input,out=y)

# Paddle 写法
paddle.assign(paddle.lgamma(input), y)
```
