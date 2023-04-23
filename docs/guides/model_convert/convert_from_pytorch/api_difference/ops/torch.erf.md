## [torch 参数更多 ]torch.erf
### [torch.erf](https://pytorch.org/docs/stable/generated/torch.erf.html?highlight=erf#torch.erf)

```python
torch.erf(input,
          *,
          out=None)
```

### [paddle.erf](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/erf_cn.html#erf)

```python
paddle.erf(x,
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
torch.erf([0, -1., 10.], out=y)

# Paddle 写法
paddle.assign(paddle.erf([0, -1., 10.]), y)
```
