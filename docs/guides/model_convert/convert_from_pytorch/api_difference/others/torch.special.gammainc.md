### [ torch 参数更多 ] torch.special.gammainc

### [torch.special.gammainc](https://pytorch.org/docs/stable/special.html#torch.special.gammainc)

```python
torch.special.gammainc(input, 
                       other,
                       *,
                       out=None)
```

### [paddle.gammainc](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.6/api/index_cn.html)

```python
paddle.gammainc(x, 
                y)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | Paddle | 备注                                                         |
| ------------- | ------ | ------------------------------------------------------------ |
| input           | x      |  第一个非负输入张量，仅参数名不一致。         |
| other           | y      | 第二个非负输入张量，仅参数名不一致。         |
| out        | -      | Paddle 无此参数，需要转写。 |

### 转写示例
#### out：指定输出
```python
# PyTorch 写法
torch.special.gammainc(torch.tensor([1.0,2.0]),torch.tensor([3.0,4.0]),out=y)

# Paddle 写法
paddle.assign(paddle.gammainc(paddle.to_tensor([1.0,2.0]),paddle.to_tensor([3.0,4.0])), y)
```