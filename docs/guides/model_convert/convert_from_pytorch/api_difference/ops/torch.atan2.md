## [torch 参数更多 ]torch.atan2
### [torch.atan2](https://pytorch.org/docs/1.13/generated/torch.atan2.html#torch.atan2)

```python
torch.atan2(input,
            other,
            *,
            out=None)
```

### [paddle.atan2](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/atan2_cn.html)

```python
paddle.atan2(x,
             y,
             name=None)
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input | x | 表示输入的 Tensor ，仅参数名不一致。  |
| other | y | 表示输入的 Tensor ，仅参数名不一致。  |
| out | -  | 表示输出的 Tensor ， Paddle 无此参数，需要进行转写。    |


### 转写示例
#### out：指定输出
```python
# Pytorch 写法
torch.atan2(torch.tensor([0.2,0.3]),torch.tensor([0.4,0.5]),out=y)

# Paddle 写法
paddle.assign(paddle.atan2(paddle.to_tensor([0.2,0.3]),paddle.to_tensor([0.4,0.5])),y)
```
