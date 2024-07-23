## [ 仅参数名不一致 ]torch.view_as_complex
### [torch.view_as_complex](https://pytorch.org/docs/stable/generated/torch.view_as_complex.html?highlight=view_as_complex#torch.view_as_complex)

```python
torch.view_as_complex(input)
```

### [paddle.as_complex](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/as_complex_cn.html#as-complex)

```python
paddle.as_complex(x,
                  name=None)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 输入的 Tensor ，仅参数名不一致。                   |
