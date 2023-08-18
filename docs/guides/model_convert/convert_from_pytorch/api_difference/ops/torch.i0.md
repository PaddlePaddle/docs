## [ torch 参数更多 ]torch.i0

### [torch.i0](https://pytorch.org/docs/stable/special.html#torch.special.i0)

```python
torch.i0(input,
         *,
         out=None)
```

### [paddle.i0](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/i0_cn.html)

```python
paddle.i0(x,
          name=None)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input |  x  | 表示输入的 Tensor，仅参数名不一致。  |
|  out  |  -  | 表示输出的 Tensor，Paddle 无此参数，需要转写。    |


### 转写示例
#### out：指定输出
```python
# Pytorch 写法
torch.i0(torch.tensor([1.0000, 1.2661, 2.2796]), out=y)

# Paddle 写法
paddle.assign(paddle.i0(paddle.to_tensor([1.0000, 1.2661, 2.2796])), y)
```
