## [ torch 参数更多 ]torch.frexp

### [torch.frexp](https://pytorch.org/docs/1.13/generated/torch.frexp.html?highlight=frexp#torch.frexp)

```python
torch.frexp(input,
            out=None)
```

### [paddle.frexp](暂无对应文档)

```python
paddle.frexp(x,
             name=None)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input          | x         | 表示输入的 Tensor，仅参数名不一致。                                     |
| out       | -        | 表示输出的 Tensor,可选项，Paddle 无此参数，需要进行转写。 |

### 转写示例

#### out：指定输出

```python
# Pytorch 写法
torch.frexp(x,out=y)

# Paddle 写法
paddle.assign(paddle.frexp(x), y)
```
