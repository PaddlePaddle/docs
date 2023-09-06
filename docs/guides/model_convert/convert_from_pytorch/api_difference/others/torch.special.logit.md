## [torch 参数更多]torch.special.logit

### [torch.special.logit](https://pytorch.org/docs/stable/special.html#torch.special.logit)

```python
torch.special.logit(input, eps=None, *, out=None)
```

### [paddle.logit](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/logit_cn.html)

```python
paddle.logit(x, eps=None, name=None)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                               |
| ------- | ------------ | -------------------------------------------------- |
| input   | x            | 输入的 Tensor，仅参数名不一致。                |
| eps     | eps          | 传入该参数后可将 x 的范围控制在 [eps,1−eps]。      |
| out     | -            | 表示输出的 Tensor，Paddle 无此参数，需要转写。 |

### 转写示例

#### out：指定输出

```python
# Pytorch 写法
torch.special.logit(x, out=y)

# Paddle 写法
paddle.assign(paddle.logit(x), y)
```
