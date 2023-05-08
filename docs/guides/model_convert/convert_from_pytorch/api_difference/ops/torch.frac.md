## [torch 参数更多 ]torch.frac
### [torch.frac](https://pytorch.org/docs/1.13/generated/torch.frac.html?highlight=frac#torch.frac)

```python
torch.frac(input,
            *,
            out=None)
```

### [paddle.frac](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/frac_cn.html#frac)

```python
paddle.frac(x,
            name=None)
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| <font color='red'> input </font> | <font color='red'> x </font> | 表示输入的 Tensor ，仅参数名不一致。  |
| <font color='red'> out </font> | -  | 表示输出的 Tensor ， Paddle 无此参数，需要进行转写。    |



### 转写示例
#### out：指定输出
```python
# Pytorch 写法
torch.frac(input, out=y)

# Paddle 写法
input1 = paddle.to_tensor(input)
paddle.assign(paddle.frac(input1), y)
```
