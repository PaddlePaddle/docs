## [ torch 参数更多 ]torch.multinomial
### [torch.multinomial](https://pytorch.org/docs/stable/generated/torch.multinomial.html#torch.multinomial)
```python
torch.multinomial(input,
                  num_samples,
                  replacement=False,
                  *,
                  generator=None,
                  out=None)
```
### [paddle.multinomial](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/multinomial_cn.html)
```python
paddle.multinomial(x,
                   num_samples=1,
                   replacement=False,
                   name=None)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
|  input              |  x           | 表示输入的 Tensor，仅参数名不一致。  |
| num_samples         | num_samples  | 表示采样的次数。                                     |
| replacement         | replacement  | 表示是否是可放回的采样。                                     |
| generator           | -            | 用于采样的伪随机数生成器，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。      |
|  out                | -            | 表示输出的 Tensor，Paddle 无此参数，需要转写。    |


### 转写示例
#### out：指定输出
```python
# Pytorch 写法
torch.multinomial(torch.tensor([0.3, 0.5, 0.2]), out=y)

# Paddle 写法
paddle.assign(paddle.multinomial(paddle.to_tensor([0.3, 0.5, 0.2])), y)
```
```
