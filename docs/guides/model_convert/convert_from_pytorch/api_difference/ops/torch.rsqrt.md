## [torch 参数更多 ]torch.rsqrt
### [torch.rsqrt](https://pytorch.org/docs/stable/generated/torch.rsqrt.html?highlight=rsqrt#torch.rsqrt)

```python
torch.rsqrt(input,
            *,
            out=None)
```

### [paddle.rsqrt](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/rsqrt_cn.html#rsqrt)

```python
paddle.rsqrt(x,
             name=None)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 输入的 Tensor ，仅参数名不一致。                                      |
| out           | -            | 表示输出的 Tensor，Paddle 无此参数，需要转写。               |


### 转写示例
#### out：指定输出
```python
# Pytorch 写法
torch.rsqrt([3, 5], out=y)

# Paddle 写法
paddle.assign(paddle.rsqrt([3, 5]), y)
```
