## [torch 参数更多 ]torch.lerp
### [torch.lerp](https://pytorch.org/docs/stable/generated/torch.lerp.html)

```python
torch.lerp(input,
          end,
          weight,
          *,
          out=None)
```

### [paddle.lerp](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/lerp_cn.html#lerp)

```python
paddle.lerp(x,
            y,
            weight,
            name=None)
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数差异
|    PyTorch        | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| <font color='red'> input </font> | <font color='red'> x </font> | 表示输入的 Tensor ，仅参数名不一致。  |
| <font color='red'> end </font> | <font color='red'> y </font> | 表示输入的 Tensor ，仅参数名不一致。  |
| <font color='red'> weight </font> | <font color='red'> weight </font> | 表示输入的 Tensor 。  |
| <font color='red'> out </font> | -  | 表示输出的 Tensor ， Paddle 无此参数，需要进行转写。    |



### 转写示例
#### out：指定输出
```python
# Pytorch 写法
torch.lerp(tensor([ 1.,  2.,  3.,  4.]),
          tensor([ 10.,  10.,  10.,  10.]),
          0.5, out=y)

# Paddle 写法
paddle.assign(paddle.lerp(tensor([ 1.,  2.,  3.,  4.]),
          tensor([ 10.,  10.,  10.,  10.]), 0.5), y)
```
