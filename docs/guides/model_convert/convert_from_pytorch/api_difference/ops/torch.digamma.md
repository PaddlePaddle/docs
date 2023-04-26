## [ torch 参数更多 ]torch.digamma
### [torch.digamma](https://pytorch.org/docs/1.13/generated/torch.digamma.html?highlight=torch+digamma#torch.digamma)
```python
torch.digamma(input,
              *,
              out=None)
```

### [paddle.digamma](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/digamma_cn.html)
```python
paddle.digamma(x,
               name=None)
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
|  input  |  x  | 表示输入的 Tensor ，仅参数名不一致。  |
|  out  | -  | 表示输出的 Tensor ， Paddle 无此参数，需要进行转写。    |

### 转写示例
#### out：指定输出
```python
# Pytorch 写法
torch.digamma(x, out=y)

# Paddle 写法
paddle.assign(paddle.digamma(x), y)
```
