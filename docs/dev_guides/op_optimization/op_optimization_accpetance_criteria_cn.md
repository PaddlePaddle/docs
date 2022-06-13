# 算子性能优化 验收规范

## CI通过性

提交至 Paddle repo 的 Pull Request（简称 PR），涉及到的相关检测CI必须全部 Pass。用来验证对之前功能点的兼容和影响，保障新合入代码对历史代码不产生影响。

新增代码必须要有相应的单测保障测试覆盖率达到准入要求（测试覆盖率（行覆盖率)90%）。

## 性能测试

性能测试建议采用OP Benchmark测试算子性能。经过性能优化，[OP Benchmark](https://github.com/PaddlePaddle/benchmark/tree/master/api)中全部case不能出现性能下降，需要通过列表，对比性能优化前后的OP性能情况。

## PR内容描述要求

单元测试内容需要和开发代码放在同一个PR提交，后续修改也需要基于此PR。PR内容描述测试部分需要明确描述下列内容：

    1. 合入前Paddle中算子的性能现状

    2. 业内最优方案的算子性能现状

    3. PR性能优化方案概述

    4. 性能优化对比表格

## OP测试内容及单元测试要求

性能测试至少覆盖[OP Benchmark](https://github.com/PaddlePaddle/benchmark/tree/master/api)中全部的case场景。OP性能优化后，需要在 Paddle 单元测试中对GPU Kernel进行有效性和边界值测试。

## 交流与改进

PR的单测部分必须 Paddle 测试人员 review，确保完整覆盖了待测功能点后，会给予 approved。如果 review 过程中发现测试缺失和遗漏的测试点，会通过 GitHub 代码行 Comment 的和 Request Changes 的方式交流改进，待PR修改完毕后给予 approved。

## 后续维护

代码成功合入后，如果发现对框架造成了性能下降影响，或者和部分功能存在严重冲突导致Bug，会对代码进行 Revert 并通过 ISSUE 告知相关的开发者，请提交 PR 修复问题，并重新合入。
