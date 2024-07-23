## [torch 参数更多 ]torch.log
### [torch.log](https://pytorch.org/docs/stable/generated/torch.log.html?highlight=log#torch.log)

```python
torch.log(input,
          *,
          out=None)
```

### [paddle.log](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/log_cn.html#log)

```python
paddle.log(x,
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
torch.log(input, out=y)

# Paddle 写法
paddle.assign(paddle.log(input), y)
```
