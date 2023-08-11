## [torch 参数更多 ]torch.log10
### [torch.log10](https://pytorch.org/docs/stable/generated/torch.log10.html?highlight=log10#torch.log10)

```python
torch.log10(input,
            *,
            out=None)
```

### [paddle.log10](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/log10_cn.html#log10)

```python
paddle.log10(x,
             name=None)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| <font color='red'> input </font> | <font color='red'> x </font> | 表示输入的 Tensor ，仅参数名不一致。  |
| <font color='red'> out </font> | -  | 表示输出的 Tensor ， Paddle 无此参数，需要转写。    |


### 转写示例
#### out：指定输出
```python
# Pytorch 写法
torch.log10(input, out=y)

# Paddle 写法
paddle.assign(paddle.log10(input), y)
```
