## [参数完全一致]transformers.modeling_outputs.BaseModelOutputWithPast

### [transformers.modeling_outputs.BaseModelOutputWithPast](https://hf-mirror.com/docs/transformers/v4.42.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast)

```python
transformers.modeling_outputs.BaseModelOutputWithPast(last_hidden_state: FloatTensor = None, past_key_values: Optional = None, hidden_states: Optional = None, attentions: Optional = None)
```

### [paddlenlp.transformers.model_outputs.BaseModelOutputWithPast](https://github.com/PaddlePaddle/PaddleNLP/blob/e336e78c338d2514ee6c937982ce5d8c960b85ff/paddlenlp/transformers/model_outputs.py#L590)

```python
paddlenlp.transformers.model_outputs.BaseModelOutputWithPast(last_hidden_state: paddle.Tensor = None, past_key_values: Optional[Tuple[Tuple[paddle.Tensor]]] = None, hidden_states: Optional[Tuple[paddle.Tensor]] = None, attentions: Optional[Tuple[paddle.Tensor]] = None)
```


功能一致，参数完全一致，具体如下：

### 参数映射

| transformers      | PaddlePaddle                                          | 备注                                                    |
| ----------------- | ----------------------------------------------------- | ------------------------------------------------------- |
| last_hidden_state | last_hidden_state                                     | 最后一个隐藏状态。默认值为 None。                        |
| past_key_values   | past_key_values                                       | 过去的键值对。默认值为 None。                           |
| hidden_states     | hidden_states                                         | 隐藏状态。默认值为 None。                               |
| attentions        | attentions                                            | 注意力权重。默认值为 None。                             |
