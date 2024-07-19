## [torch 参数更多]transformers.GenerationConfig

### [transformers.GenerationConfig](https://hf-mirror.com/docs/transformers/v4.42.0/en/main_classes/text_generation#transformers.GenerationConfig)

```python
transformers.GenerationConfig(*kwargs)
```

### [paddlenlp.generation.GenerationConfig](https://github.com/PaddlePaddle/PaddleNLP/blob/e336e78c338d2514ee6c937982ce5d8c960b85ff/paddlenlp/generation/configuration_utils.py#L62)

```python
paddlenlp.generation.GenerationConfig(*kwargs)
```

两者功能一致，但 PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| transformers                         | PaddlePaddle        | 备注      |
| -------------------------------------| ------------------- | -------- |
| max_length                           | max_length          | 最大生成长度。 |
| max_new_tokens                       | -                   | 最大生成长度(忽略 promot)，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。|
| min_length                           | min_length          | 最小生成长度。 |
| min_new_tokens                       | -                   | 最小生成长度(忽略 promot)，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
| early_stopping                       | early_stopping      | 早停是否开启。 |
| max_time                             | -                   | 最大允许计算运行时间，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
| do_sample                            | do_sample           | 是否进行采样。 |
| num_beams                            | num_beams           | beams for beam search。 |
| num_beam_groups                      | num_beam_groups     | beams 划分的组数。 |
| penalty_alpha                        | -                   | 惩罚参数，Paddle 无此参数，暂无转写方式。 |
| use_cache                            | use_cache           | 是否开启 kv cache。 |
| temperature                          | temperature         | 用于控制下个 token 生成的参数。 |
| top_k                                | top_k               | top_k 算法的 k 值。 |
| top_p                                | top_p               | top_p 算法的 p 值。 |
| typical_p                            | -                   | 局部典型度量的参数，Paddle 无此参数，暂无转写方式。 |
| epsilon_cutoff                       | -                   | 截断采样参数，Paddle 无此参数，暂无转写方式。 |
| eta_cutoff                           | -                   | 截断采样参数，Paddle 无此参数，暂无转写方式。 |
| diversity_penalty                    | -                   | 分散惩罚系数，Paddle 无此参数，暂无转写方式。 |
| repetition_penalty                   | repetition_penalty  | 重复惩罚参数。 |
| encoder_repetition_penalty           | -                   | 编码重复惩罚参数，Paddle 无此参数，暂无转写方式。 |
| length_penalty                       | length_penalty      | 长度重复惩罚参数。 |
| no_repeat_ngram_size                 | -                   | ngram 在给定长度内不可重复，Paddle 无此参数，暂无转写方式。 |
| bad_words_ids                        | -                   | 不允许生成的 id 列表，Paddle 无此参数，暂无转写方式。 |
| force_words_ids                      | -                   | 必须生成的 id 列表，Paddle 无此参数，暂无转写方式。 |
| renormalize_logits                   | -                   | 对 logits 进行 renormalize 操作，Paddle 无此参数，暂无转写方式。  |
| constraints                          | -                   | 自定义约束列表，Paddle 无此参数，暂无转写方式。 |
| forced_bos_token_id                  | forced_bos_token_id | 强制生成的 bos token 的 id。 |
| forced_eos_token_id                  | forced_eos_token_id | 强制生成的 eos token 的 id。 |
| remove_invalid_values                | -                   | 是否移除无效值，Paddle 无此参数，暂无转写方式。 |
| exponential_decay_length_penalty     | -                   | 生成指定长度 tokens 后的惩罚参数，Paddle 无此参数，暂无转写方式。 |
| suppress_tokens                      | -                   | 生成期间被抑制的 tokens，Paddle 无此参数，暂无转写方式。 |
| begin_suppress_tokens                | -                   | 开始生成时被抑制的 tokens，Paddle 无此参数，暂无转写方式。 |
| forced_decoder_ids                   | -                   | 指定 decoder 指定位置生成的 token，Paddle 无此参数，暂无转写方式。 |
| sequence_bias                        | -                   | 映射 sequence 到其偏执项的字典，Paddle 无此参数，暂无转写方式。 |
| guidance_scale                       | -                   | 控制 output 与 input 联系紧密程度的参数，Paddle 无此参数，暂无转写方式。 |
| low_memory                           | -                   | 使用更低显存占用的搜索算法，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
| num_return_sequences                 | -                   | 为 batch 中每个序列独立计算返回序列的个数，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。|
| output_attentions                    | -                   | 是否返回注意力层的注意力张量，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
| output_hidden_states                 | -                   | 是否返回注意力层的隐藏层张量，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
| output_scores                        | -                   | 是否返回注意力层的得分张量，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
| return_dict_in_generate              | -                   | 是否返回 dict 而不是 tuple，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。  |
| pad_token_id                         | pad_token_id        | padding token 的 id。 |
| bos_token_id                         | bos_token_id        | beginning-of-sequence token 的 id。 |
| eos_token_id                         | eos_token_id        | end-of-sequence  token 的 id。 |
| encoder_no_repeat_ngram_size         | -                   | encoder 中指定 size 内不能出现同一个 ngram，Paddle 无此参数，暂无转写方式。 |
| decoder_start_token_id               | -                   | decoder 生成的第一个 token 的 id，Paddle 无此参数，暂无转写方式。 |
| num_assistant_tokens                 | -                   | 定义在每次迭代中由目标模型检查之前由辅助模型生成的推测令牌的数量，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
| num_assistant_tokens_schedule        | -                   | 定义推理时应更改最大辅助 tokens 的 schedule，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
