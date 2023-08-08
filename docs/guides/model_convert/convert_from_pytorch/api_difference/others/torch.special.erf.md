## [ torch 参数更多 ]torch.special.erf

### [torch.special.erf](https://pytorch.org/docs/stable/special.html?highlight=torch+special+erf#torch.special.erf)

```python
torch.special.erf(input,
                  out=None)
```

### [paddle.erf](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/erf_cn.html)

```python
paddle.erf(x,
          name=None)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input          | x         | 表示输入的 Tensor ，仅参数名不一致。                                     |
| out        | -        | 表示输出的 Tensor ， Paddle 无此参数，需要转写。 |

### 转写示例

#### out：指定输出

```python
# Pytorch 写法
torch.special.erf(t, out=y)

# Paddle 写法
paddle.assign(paddle.erf(t), y)
```
