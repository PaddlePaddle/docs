## [torch 参数更多 ]torch.log1p
### [torch.log1p](https://pytorch.org/docs/stable/generated/torch.log1p.html?highlight=log1p#torch.log1p)

```python
torch.log1p(input,
            *,
            out=None)
```

### [paddle.log1p](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/log1p_cn.html#log1p)

```python
paddle.log1p(x,
             name=None)
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| <font color='red'> input </font> | <font color='red'> x </font> | 表示输入的 Tensor ，仅参数名不一致。  |
| <font color='red'> out </font> | -  | 表示输出的 Tensor ， Paddle 无此参数，需要进行转写。    |


### 转写示例
#### out：指定输出
```python
# Pytorch 写法
torch.log1p([3, 5], out=y)

# Paddle 写法
paddle.assign(paddle.log1p([3, 5]), y)
```
