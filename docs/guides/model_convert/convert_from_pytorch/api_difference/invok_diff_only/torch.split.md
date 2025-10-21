## [ 仅 API 调用方式不一致 ]torch.split
### [torch.split](https://pytorch.org/docs/stable/generated/torch.split.html?highlight=torch%20split#torch.split)
```python
torch.split(tensor,
            split_size_or_sections,
            dim=0)
```

### [paddle.split](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/split_cn.html#split)
```python
paddle.split(x,
             num_or_sections,
             axis=0,
             name=None)
```

其中 PyTorch 的 `split_size_or_sections` 与 Paddle 的 `num_or_sections` 用法不一致，具体如下：
