# 中文API文档贡献指南


PaddlePaddle 的中文API文档以 rst 文件的格式，存储于 [PaddlePaddle/FluidDoc](https://github.com/PaddlePaddle/FluidDoc) 中，通过技术手段，将rst文件转为 HTML文件后呈现至[官网API文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/index_cn.html) 。如果想要修改中文API文档，需要按以下流程完成修改。

## 一、修改前的准备工作

### 1.1 Fork
先跳转到  [PaddlePaddle/FluidDoc](https://github.com/PaddlePaddle/FluidDoc) GitHub 首页，然后单击 Fork 按钮，生成自己仓库下的目录，比如你的 GitHub 用户名为 USERNAME，则生成： https://github.com/USERNAME/FluidDoc。

### 1.2 Clone
将你目录下的远程仓库clone到本地。
```
➜ git clone https://github.com/USERNAME/FluidDoc
➜ cd FluidDoc
```

### 1.3 创建本地分支

FluidDoc 目前使用 [Git流分支模型](https://nvie.com/posts/a-successful-git-branching-model/)进行开发，测试，发行和维护。

所有的 feature 和 bug fix 的开发工作都应该在一个新的分支上完成，一般从 develop 分支上创建新分支。

使用 ``git checkout -b`` 创建并切换到新分支。

```
➜  git checkout -b my-cool-stuff
```
值得注意的是，在 ``checkout`` 之前，需要保持当前分支目录 clean，否则会把 untracked 的文件也带到新分支上，这可以通过 ``git status`` 查看。

### 1.4 下载 pre-commit 钩子工具（若有的话，可以跳过此步骤）

Paddle 开发人员使用 [pre-commit](https://pre-commit.com/) 工具来管理 Git 预提交钩子。 它可以帮助你格式化源代码（C++，Python），在提交（commit）前自动检查一些基本事宜（如每个文件只有一个 EOL，Git 中不要添加大文件等）。

pre-commit测试是 Travis-CI 中单元测试的一部分，不满足钩子的 PR 不能被提交到 Paddle，首先安装并在当前目录运行它：

```
➜  pip install pre-commit
➜  pre-commit install
```

Paddle 使用 clang-format 来调整 C/C++ 源代码格式，请确保 clang-format 版本在 3.8 以上。

**注**：通过``pip install pre-commit``和 ``conda install -c conda-forge pre-commit``安装的yapf稍有不同，Paddle 开发人员使用的是 ``pip install pre-commit``。

## 二、正式修改API文档

目前，[FluidDoc](https://github.com/PaddlePaddle/FluidDoc) 的 `doc/paddle/api` 下存放了与 Paddle 中文API文档所有相关的文件。说明如下：
```
doc/paddle/api
|--paddle                      # 存放中文API文档，文件名为api_name_cn.rst,路径为真实实现的路径
|    |--amp
|    |--compat
|    |--device
...
|    |--utils
|    |--vision
|-- alias_api_mapping           # 定义了API别名关系，影响官网API文档展示逻辑
|-- api_label                   # 英文API文档的标签，用于API文档的相互引用
|-- display_doc_list  
|-- gen_alias_api.py            # 生成全量的API别名关系
|-- gen_alias_mapping.sh        # 已废弃
|-- gen_doc.py                  # 生成英文API文档目录树程序
|-- gen_doc.sh                  # 生成英文API文档目录树脚本
|-- index_cn.rst                # 官网中文API文档首页
|-- index_en_rst                # 官网英文API文档首页
|-- not_display_doc_list        # 官网不展示的API列表
```

### 2.1 新增 API 文档

当你新增了一个API时，需要同时新增该API的中文文档。你需要在该API文档的真实实现路径下，新建一个 api_name_cn.rst 文件，文件内容需要按照 [飞桨API文档书写规范](https://github.com/PaddlePaddle/FluidDoc/wiki/%E9%A3%9E%E6%A1%A8API%E6%96%87%E6%A1%A3%E4%B9%A6%E5%86%99%E8%A7%84%E8%8C%83)进行书写。

**注意：** 真实实现路径是指，该API在 [PaddlePaddle/Paddle](https://github.com/PaddlePaddle/Paddle) 的实现路径。举例来说，对于 ``paddle.all``，其实现的路径为 ``python/paddle/tensor/math.py``，因此其真实实现的路径为 ``paddle/tensor/math``；所以，在 [PaddlePaddle/FluidDoc](https://github.com/PaddlePaddle/FluidDoc) 中，需要在 ``doc/paddle/api`` 目录下，进入``paddle/tensor/math``，然后新建 all_cn.rst 文件，按 [规范](https://github.com/PaddlePaddle/FluidDoc/wiki/%E9%A3%9E%E6%A1%A8API%E6%96%87%E6%A1%A3%E4%B9%A6%E5%86%99%E8%A7%84%E8%8C%83) 书写``paddle.all`` 的中文文档内容即可。

**注意：** Paddle 中存在部分特殊的API：[飞桨特殊API实现](https://github.com/PaddlePaddle/FluidDoc/wiki/%E9%A3%9E%E6%A1%A8API%E7%89%B9%E6%AE%8A%E5%AE%9E%E7%8E%B0)，其真实实现不能按如上的规则确定，可查看  [飞桨特殊API实现](https://github.com/PaddlePaddle/FluidDoc/wiki/%E9%A3%9E%E6%A1%A8API%E7%89%B9%E6%AE%8A%E5%AE%9E%E7%8E%B0) 确认其真实实现的路径。

### 2.2 修改 API 文档

修改中文API文档，可以通过API的URL，确定API文档的源文件。如 ``paddle.all`` 的中文API文档URL为：[https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/tensor/math/all_cn.html](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/tensor/math/all_cn.html)，URL路径中，``api/paddle/tensor/math/all_cn.html`` 即对应 ``(FluidDoc/doc/paddle/)api/paddle/tensor/math/all_cn.rst`` , 因此，可以很快的确定中文API文档的源文件，然后直接修改即可。

### 2.3 修改API别名关系
Paddle 的API目前存在别名关系，如同一个API ``all`` 的调用方式有三种，分别为``paddle.all、paddle.tensor.all、paddle.tensor.math.all``；其中，由于paddle 建议使用路径较短的API，因此 ``paddle.all`` 即为 ``all`` 的推荐别名；又因为 ``all`` 的实现在 [PaddlePaddle/Paddle](https://github.com/PaddlePaddle/Paddle) 的 ``python/paddle/tensor/math.py`` 中，因此其真实实现的别名为 ``paddle.tensor.math.all``。

而[飞桨官网API文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/index_cn.html)针对每一个API，都仅在推荐别名目录下展示其API文档，如只在 ``paddle`` 目录下展示 ``all`` 的API文档，而不会在 ``paddle.tensor`` 与 ``paddle.tensor.math`` 。

这个逻辑的实现依赖 [alias_api_mapping](https://github.com/PaddlePaddle/FluidDoc/blob/develop/doc/paddle/api/alias_api_mapping)，这个文件定义了Paddle API 的别名关系，每一行的规则为API的 **真实实现 + '\t' + 推荐别名,别名1,别名2...** ；
如对于 ``all`` ，其别名关系为

    paddle.tensor.math.all    paddle.all,paddle.tensor.all

就表明 ``all`` 这个API的真实实现为 ``paddle.tensor.math.all``，但是推荐别名为 ``paddle.all`` 。

**注意：** 对于一些特殊的API，具体列表见：[飞桨特殊API实现](https://github.com/PaddlePaddle/FluidDoc/wiki/%E9%A3%9E%E6%A1%A8API%E7%89%B9%E6%AE%8A%E5%AE%9E%E7%8E%B0),该列表中给出了每个API的真实实现与推荐别名。

## 三、提交&push


### 3.1 提交&触发CI单测

- 修改 ``paddle/tensor/math/all_cn.rst`` 这个文件，并提交这个文件

   ```
   ➜  git status
   On branch my-cool-stuff
  Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git restore <file>..." to discard changes in working directory)
  modified:   paddle/tensor/math/all_cn.rst

  no changes added to commit (use "git add" and/or "git commit -a")

  ➜  git add  paddle/tensor/math/all_cn.rst
   ```

  **如果你不想提交本次修改**，使用 ``git checkout -- <file>`` 取消上面对``python/paddle/tensor/math.py``文件的提交，可以将它恢复至上一次提交的状态:
   ```
  ➜  git checkout  -- paddle/tensor/math/all_cn.rst
   ```
   恢复后重新进行修改并提交文件即可。

- pre-commit：提交修改说明前，需要对本次修改做一些格式化检查：

    ```
    ➜  pre-commit
    CRLF end-lines remover...............................(no files to check)Skipped
    yapf.....................................................................Passed
    Check for added large files..............................................Passed
    Check for merge conflicts................................................Passed
    Check for broken symlinks................................................Passed
    Detect Private Key...................................(no files to check)Skipped
    Fix End of Files.........................................................Passed
    clang-format.........................................(no files to check)Skipped
    cpplint..............................................(no files to check)Skipped
    pylint...................................................................Passed
    copyright_checker........................................................Passed
    ```
  全部Passed 或 Skipped后，即可进入下一步。如果有 Failed 文件，则需要按照规范，修改出现Failed 的文件后，重新 ``git add -> pre-commit`` ，直至没有 Failed 文件。

    ```
    ➜  pre-commit
    CRLF end-lines remover...............................(no files to check)Skipped
    yapf.....................................................................Failed
    - hook id: yapf
    - files were modified by this hook
    Check for added large files..............................................Passed
    Check for merge conflicts................................................Passed
    Check for broken symlinks................................................Passed
    Detect Private Key...................................(no files to check)Skipped
    Fix End of Files.........................................................Passed
    clang-format.........................................(no files to check)Skipped
    cpplint..............................................(no files to check)Skipped
    pylint...................................................................Failed
    - hook id: pylint-doc-string
    - exit code: 127

    ./tools/codestyle/pylint_pre_commit.hook: line 11: pylint: command not found

    copyright_checker........................................................Passed
    ```
- 填写提交说明：Git 每次提交代码，都需要写提交说明，让其他人知道这次提交做了哪些改变，可以通过 ``git commit`` 完成：

   ```
   ➜  git commit -m "fix all docs bugs"
   ```

### 3.2 确保本地仓库是最新的

在准备发起 Pull Request 之前，需要同步原仓库（https://github.com/PaddlePaddle/FluidDoc）最新的代码。

首先通过 ``git remote`` 查看当前远程仓库的名字。

```
➜  git remote
origin
➜  git remote -v
origin    https://github.com/USERNAME/FluidDoc (fetch)
origin    https://github.com/USERNAME/FluidDoc (push)
```

这里 origin 是你 clone 的远程仓库的名字，也就是自己用户名下的 Paddle，接下来创建一个原始 Paddle 仓库的远程主机，命名为 upstream。

```
➜  git remote add upstream https://github.com/PaddlePaddle/FluidDoc
➜  git remote
origin
upstream
```

获取 upstream 的最新代码并更新当前分支。
```
➜  git fetch upstream
➜  git pull upstream develop
```


### 3.3 Push 到远程仓库

将本地的修改推送到 GitHub 上，也就是 https://github.com/USERNAME/FluidDoc。

```
# 推送到远程仓库 origin 的 my-cool-stuff 分支上
➜  git push origin my-cool-stuff
```

## 四、提交PR

在你push后在对应仓库会提醒你进行PR操作：

点击后，按格式填写PR内容，即可。

## 五、review&merge

提交PR后，可以点击右侧的Reviewers，指定 Paddle 的同学进行 Review。

目前，Paddle 负责API文档的同学是 @TCChenLong，可以直接指定他进行文档Review；此外，你也可以指定 @jzhang533、@saxon-zh、@Heeenrrry、@swtkiwi、@dingjiaweiww等同学review 。

## CI

Paddle 中与文档相关的CI 流水线是 `PR-CI-CPU-Py2-18`等，主要对以下几个方面进行检查:

- 检查PR CLA
- 检查增量修改的API是否需要相关人员审核
- 若需要执行示例代码则执行看能否正常运行

如果无法通过该CI，请点击对应CI的details，查看CI运行的的log，并根据log修改你的PR，直至通过CI。
