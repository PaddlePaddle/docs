# 中文 API 文档贡献指南


PaddlePaddle 的中文 API 文档以 rst 文件的格式，存储于 [PaddlePaddle/docs](https://github.com/PaddlePaddle/docs) 中，通过技术手段，将 rst 文件转为 HTML 文件后呈现至[官网 API 文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/index_cn.html) 。如果想要修改中文 API 文档，需要按以下流程完成修改。


## 一、修改前的准备工作

### 1.1 Fork
先跳转到  [PaddlePaddle/docs](https://github.com/PaddlePaddle/docs) GitHub 首页，然后单击 Fork 按钮，生成自己仓库下的目录，比如你的 GitHub 用户名为 USERNAME，则生成： https://github.com/USERNAME/docs。

### 1.2 Clone
将你目录下的远程仓库 clone 到本地。
```
➜ git clone https://github.com/USERNAME/docs
➜ cd docs
```

### 1.3 创建本地分支

docs 目前使用 [Git 流分支模型](https://nvie.com/posts/a-successful-git-branching-model/)进行开发，测试，发行和维护。

所有的 feature 和 bug fix 的开发工作都应该在一个新的分支上完成，一般从 develop 分支上创建新分支。

使用 ``git checkout -b`` 创建并切换到新分支。

```
➜  git checkout -b my-cool-stuff
```
值得注意的是，在 ``checkout`` 之前，需要保持当前分支目录 clean，否则会把 untracked 的文件也带到新分支上，这可以通过 ``git status`` 查看。

### 1.4 下载 pre-commit 钩子工具（若有的话，可以跳过此步骤）

Paddle 开发人员使用 [pre-commit](https://pre-commit.com/) 工具来管理 Git 预提交钩子。 它可以帮助你格式化源代码（C++，Python），在提交（commit）前自动检查一些基本事宜（如每个文件只有一个 EOL，Git 中不要添加大文件等）。

pre-commit 测试是 Travis-CI 中单元测试的一部分，不满足钩子的 PR 不能被提交到 Paddle，首先安装并在当前目录运行它：

```
➜  pip install pre-commit
➜  pre-commit install
```

Paddle 使用 clang-format 来调整 C/C++ 源代码格式，请确保 clang-format 版本在 3.8 以上。

**注**：通过``pip install pre-commit``和 ``conda install -c conda-forge pre-commit``安装的 yapf 稍有不同，Paddle 开发人员使用的是 ``pip install pre-commit``。

## 二、正式修改 API 文档

目前，[docs](https://github.com/PaddlePaddle/docs) 的 `docs/api/` 下存放了与 Paddle 中文 API 文档所有相关的文件。说明如下：
```
docs/api
|--paddle                      # 存放中文 API 文档，文件名为 api_name_cn.rst,路径为暴露的路径
|    |--amp
|    |--device
...
|    |--utils
|    |--vision
|-- api_label                   # 英文 API 文档的标签，用于 API 文档的相互引用
|-- display_doc_list
|-- gen_alias_api.py            # 生成全量的 API 别名关系
|-- gen_alias_mapping.sh        # 已废弃
|-- gen_doc.py                  # 生成英文 API 文档目录树程序
|-- gen_doc.sh                  # 生成英文 API 文档目录树脚本
|-- index_cn.rst                # 官网中文 API 文档首页
|-- index_en_rst                # 官网英文 API 文档首页
|-- not_display_doc_list        # 官网不展示的 API 列表
```

### 2.1 新增 API 文档

当你新增了一个 API 时，需要同时新增该 API 的中文文档。你需要在该 API 文档的暴露路径下，新建一个 api_name_cn.rst 文件，文件内容需要按照 [飞桨 API 文档书写规范](https://github.com/PaddlePaddle/docs/wiki/%E9%A3%9E%E6%A1%A8API%E6%96%87%E6%A1%A3%E4%B9%A6%E5%86%99%E8%A7%84%E8%8C%83)进行书写。

**注意：** 暴露路径是指，在开发 API 时，确认的 API 路径，如 `paddle.add`、`paddle.nn.Conv2D` 等。

### 2.2 修改 API 文档

修改中文 API 文档，可以通过 API 的 URL，确定 API 文档的源文件。如 ``paddle.all`` 的中文 API 文档 URL 为：https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/all_cn.html，URL 路径中，``api/paddle/all_cn.html`` 即对应 ``(docs/docs/)api/paddle/all_cn.rst`` , 因此，可以很快的确定中文 API 文档的源文件，然后直接修改即可。


## 三、提交&push


### 3.1 提交&触发 CI 单测

- 修改 ``paddle/all_cn.rst`` 这个文件，并提交这个文件

```
➜  git status
On branch my-cool-stuff
Changes not staged for commit:
(use "git add <file>..." to update what will be committed)
(use "git restore <file>..." to discard changes in working directory)
modified:   paddle/all_cn.rst

no changes added to commit (use "git add" and/or "git commit -a")

➜  git add  paddle/all_cn.rst
```

  **如果你不想提交本次修改**，使用 ``git checkout -- <file>`` 取消上面对``paddle/all_cn.rst``文件的提交，可以将它恢复至上一次提交的状态:
```
➜  git checkout  -- paddle/all_cn.rst
```
   恢复后重新进行修改并提交文件即可。

- pre-commit：提交修改说明前，需要对本次修改做一些格式化检查：

```
➜  pre-commit
yapf.................................................(no files to check)Skipped
Check for merge conflicts................................................Passed
Check for broken symlinks................................................Passed
Detect Private Key...................................(no files to check)Skipped
Fix End of Files.....................................(no files to check)Skipped
Trim Trailing Whitespace.............................(no files to check)Skipped
CRLF end-lines checker...............................(no files to check)Skipped
CRLF end-lines remover...............................(no files to check)Skipped
No-tabs checker......................................(no files to check)Skipped
Tabs remover.........................................(no files to check)Skipped
convert jinja2 into html.............................(no files to check)Skipped
convert-markdown-into-html...........................(no files to check)Skipped
```
  全部 Passed 或 Skipped 后，即可进入下一步。如果有 Failed 文件，则需要按照规范，修改出现 Failed 的文件后，重新 ``git add -> pre-commit`` ，直至没有 Failed 文件。
```
➜  pre-commit
yapf.................................................(no files to check)Skipped
Check for merge conflicts................................................Passed
Check for broken symlinks................................................Passed
Detect Private Key.......................................................Passed
Fix End of Files.........................................................Failed
- hook id: end-of-file-fixer
- exit code: 1
- files were modified by this hook
Trim Trailing Whitespace.................................................Passed
CRLF end-lines checker...................................................Passed
CRLF end-lines remover...................................................Passed
No-tabs checker..........................................................Passed
Tabs remover.............................................................Passed
convert jinja2 into html.................................................Passed
convert-markdown-into-html...............................................Passed
```
- 填写提交说明：Git 每次提交代码，都需要写提交说明，让其他人知道这次提交做了哪些改变，可以通过 ``git commit`` 完成：

```
➜  git commit -m "fix all docs bugs"
```

### 3.2 确保本地仓库是最新的

在准备发起 Pull Request 之前，需要同步原仓库（https://github.com/PaddlePaddle/docs）最新的代码。

首先通过 ``git remote`` 查看当前远程仓库的名字。

```
➜  git remote
origin
➜  git remote -v
origin  https://github.com/USERNAME/docs (fetch)
origin  https://github.com/USERNAME/docs (push)
```

这里 origin 是你 clone 的远程仓库的名字，也就是自己用户名下的 Paddle，接下来创建一个原始 Paddle 仓库的远程主机，命名为 upstream。

```
➜  git remote add upstream https://github.com/PaddlePaddle/docs
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

将本地的修改推送到 GitHub 上，也就是 https://github.com/USERNAME/docs。

```
# 推送到远程仓库 origin 的 my-cool-stuff 分支上
➜  git push origin my-cool-stuff
```

## 四、提交 PR

在你 push 后在对应仓库会提醒你进行 PR 操作，按格式填写 PR 内容，即可。


## 五、review&merge

提交 PR 后，可以指定 Paddle 的同学进行 Review。 目前，Paddle 负责 API 文档的同学是 @TCChenLong、@jzhang533、@saxon-zh、@Heeenrrry、@dingjiaweiww 等 。


## CI

Paddle 中与文档相关的 CI 流水线是 `Docs-NEW`等，主要对以下几个方面进行检查:

- 检查 PR CLA
- 检查增量修改的 API 是否需要相关人员审核
- 若需要执行示例代码则执行看能否正常运行

如果无法通过该 CI，请点击对应 CI 的 details，查看 CI 运行的的 log，并根据 log 修改你的 PR，直至通过 CI。
