## [ torch 参数更多 ]torch.sub
### [torch.sub](https://pytorch.org/docs/stable/generated/torch.sub.html?highlight=torch%20sub#torch.sub)

```python
torch.sub(input,
          other,
          *,
          alpha=1,
          out=None)
```

### [paddle.subtract](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/subtract_cn.html#subtract)

```python
paddle.subtract(x,
                y,
                name=None)
```

功能一致，torch 参数更多，具体如下：
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 表示被减数的 Tensor，仅参数名不一致。  |
| other         | y            | 表示减数的 Tensor，仅参数名不一致。  |
| alpha         | -            | 表示`other`的乘数，PaddlePaddle 无此参数。  |
| out           | -            | 表示输出的 Tensor，PaddlePaddle 无此参数。  |


### 转写示例
#### out：指定输出
```python
# Pytorch 写法
torch.sub(input, other, alpha=2, out=z)

# Paddle 写法
paddle.assign(paddle.subtract(x, 2*y), z)
```
