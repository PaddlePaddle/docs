## [torch 参数更多 ]torch.clone

### [torch.clone](https://pytorch.org/docs/stable/generated/torch.clone.html?highlight=clone#torch.clone)

```python
torch.clone(input,
            *,
            memory_format=torch.preserve_format)
```

### [paddle.clone](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/clone_cn.html#clone)

```python
paddle.clone(x,
             name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 输入的 Tensor ，仅参数名不一致。                          |
| memory_format | -            | 返回张量的所需内存格式，默认为 torch.preserve_format 。Paddle 无此参数，一般对训练结果影响不大，直接删除即可。            |
