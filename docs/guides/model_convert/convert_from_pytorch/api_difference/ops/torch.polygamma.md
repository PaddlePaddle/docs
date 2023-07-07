## [ torch 参数更多 ]torch.polygamma

### [torch.polygamma](https://pytorch.org/docs/2.0/special.html#torch.special.polygamma)

```python
torch.polygamma(n,
                input,
                *,
                out=None)
```

### [paddle.polygamma](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/polygamma_cn.html)

```python
paddle.polygamma(x,
                 n,
                 name=None)
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
|   n   |  n  | 指定需要求解 n 阶多伽马函数。  |
| input |  x  | 表示输入的 Tensor ，仅参数名不一致。  |
|  out  |  -  | 表示输出的 Tensor，Paddle 无此参数，需要进行转写。    |


### 转写示例
#### out：指定输出
```python
# Pytorch 写法
torch.polygamma(torch.tensor([1, 0.5]), 1, out=y)

# Paddle 写法
paddle.assign(paddle.polygamma(paddle.to_tensor([1, 0.5]), 1), y)
```
