# 算子性能优化 测试及验收规范

## CI通过性

进入PaddlePaddle主库的代码，涉及到的相关检测CI必须全部Pass。用来验证对之前功能点的兼容和影响，用来保障新合入代码对历史代码不产生影响。

新增代码必须要有相应的单测保障测试覆盖率达到准入要求（测试覆盖率（行覆盖率)90%）。

## 性能测试

性能测试建议采用OP Benchmark测试算子性能。经过性能优化，[OP Benchmark](https://github.com/PaddlePaddle/benchmark/tree/master/api)中全部case不能出现性能下降，需要列表对比性能优化前后的OP性能情况。

## PR内容描述要求

单元测试内容需要和开发代码放在同一个PR提交，后续修改也需要基于此PR。PR内容描述测试部分需要明确描述下列内容：

    1. 合入前Paddle中OP的性能现状；
    
    2. 业内最优方案的OP性能现状；

    3. PR性能优化方案概述；
    
    4. 性能优化对比表格

## OP测试内容及单元测试要求

性能测试至少覆盖[OP Benchmark](https://github.com/PaddlePaddle/benchmark/tree/master/api)中全部的case场景。OP性能优化后，需要在Paddle单元测试中对GPU Kernel进行有效性和边界值测试。

## 交流与改进

提测代码的单测部分必须paddlepaddle测试人员review，确保完整覆盖了待测功能点后，会给予approved。如果review过程中发现测试缺失和遗漏的测试点，会通过github代码行comment的和request changes的方式交流改进，待PR修改完毕后给予approved。

## 后续维护

代码成功merge后，如果发现对框架造成了性能下降影响，或者和部分功能存在严重冲突导致Bug，会对代码进行Revert并通知贡献者。待对PR修复后重新合入。