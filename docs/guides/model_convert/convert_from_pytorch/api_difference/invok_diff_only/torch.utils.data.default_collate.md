## [ 仅 API 调用方式不一致 ]torch.utils.data.default_collate
### [torch.utils.data.default_collate](https://pytorch.org/docs/stable/data.html?highlight=default_collate#torch.utils.data.default_collate)
```python
torch.utils.data.default_collate(batch)
```

### [paddle.io.dataloader.collate.default_collate_fn]()
```python
paddle.io.dataloader.collate.default_collate_fn(batch)
```

返回参数类型不一致，需要转写。具体如下：
