## [ torch 参数更多 ]torch.logit

### [torch.logit](https://pytorch.org/docs/1.13/generated/torch.logit.html?highlight=torch+logit#torch.logit)

```python
torch.logit(input, eps=None, *, out=None)
```

### [paddle.logit](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/logit_cn.html)

```python
paddle.logit(x, eps=None, name=None)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch                             | PaddlePaddle | 备注                                                                    |
| ----------------------------------- | ------------ | ----------------------------------------------------------------------- |
| input     | x           | 表示输入的 Tensor ，仅参数名不一致。                         |
| eps     | eps           | 将输入向量的范围控制在 [eps,1−eps]                        |
| out           | -      | 表示输出的 Tensor ， Paddle 无此参数，需要进行转写。         |

#### out：指定输出
```python
# Pytorch 写法
torch.logit(x, out=y)

# Paddle 写法
paddle.assign(paddle.logit(x),y)
```
