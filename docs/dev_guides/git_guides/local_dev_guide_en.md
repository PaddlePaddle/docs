# Guide of local development

You will learn how to develop programs in local environment under the guidelines of this document.

## Requirements of coding
- Please refer to the coding comment format of [Doxygen](http://www.doxygen.nl/)
- Unit test is needed for all codes.
- Pass through all unit tests.
- Please follow [regulations of submitting codes](./code_review_cn.html).

The following guidiance tells you how to submit code.
## [Fork](https://help.github.com/articles/fork-a-repo/)

Transfer to the home page of GitHub [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) ,and then click button `Fork`  to generate the git under your own file directory,such as <https://github.com/USERNAME/Paddle>。

## Clone

Clone remote git to local:

```bash
➜  git clone https://github.com/USERNAME/Paddle
➜  cd Paddle
```


## Create local branch

At present [Git stream branch model](http://nvie.com/posts/a-successful-git-branching-model/)  is applied to Paddle to undergo task of development,test,release and maintenance.Please refer to [branch regulation of Paddle](https://github.com/PaddlePaddle/docs/blob/develop/docs/design/others/releasing_process.md) about details。

All development tasks of feature and bug fix should be finished in a new branch which is extended from `develop` branch.

Create and switch to a new branch with command `git checkout -b`.


```bash
➜  git checkout -b my-cool-stuff
```

It is worth noting that before the checkout, you need to keep the current branch directory clean, otherwise the untracked file will be brought to the new branch, which can be viewed by  `git status` .


## Use `pre-commit` hook

Paddle developers use the [pre-commit](http://pre-commit.com/) tool to manage Git pre-commit hooks. It helps us format the source code (C++, Python) and automatically check some basic things before committing (such as having only one EOL per file, not adding large files in Git, etc.).

The `pre-commit` test is part of the unit test in CI. A PR that does not satisfy the hook cannot be submitted to Paddle. The pre-commit used by Paddle is version 1.10.4. Install `pre-commit` first and then run it in current directory：


```bash
➜  pip install pre-commit==1.10.4
➜  pre-commit install
```

Paddle modify the format of C/C++ source code with `clang-format` .Make sure the version of `clang-format` is above 3.8.

Note：There are differences between the installation of `yapf` with `pip install pre-commit` and that with `conda install -c conda-forge pre-commit` . Paddle developers use `pip install pre-commit` 。

## Start development

I delete a line of README.md and create a new file in the case.

View the current state via `git status` , which will prompt some changes to the current directory, and you can also view the file's specific changes via `git diff` .


```bash
➜  git status
On branch test
Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git checkout -- <file>..." to discard changes in working directory)
    modified:   README.md
Untracked files:
  (use "git add <file>..." to include in what will be committed)
    test
no changes added to commit (use "git add" and/or "git commit -a")
```

## Build

Please refer to [Compile From Source Code](../../../install/compile/fromsource_en.html) about more information of building PaddlePaddle source codes.

## Test

Any new unit testing file of the format `test_*.py`  added to the directory `python/paddle/fluid/tests/unittests/` is automatically added to the project to compile.

Remarks:

- **running unit tests requires compiling the entire project** and requires compiling with flag `WITH_TESTING` on i.e. `cmake paddle_dir -DWITH_TESTING=ON`.

- **To execute a single test, you must use the ctest command**, <font color="#FF0000">not use `python test_*.py`</font>

After successfully compiling the project, run the following command to run unit tests:

```bash
ctest -R test_mul_op -V
```

## Commit

Next we cancel the modification of README.md,and submit new added test file.

```bash
➜  git checkout -- README.md
➜  git status
On branch test
Untracked files:
  (use "git add <file>..." to include in what will be committed)
    test
nothing added to commit but untracked files present (use "git add" to track)
➜  git add test
```

It's required that the commit message is also given on every Git commit, through which other developers will be notified of what changes have been made. Type `git commit` to realize it.

```bash
➜  git commit
CRLF end-lines remover...............................(no files to check)Skipped
yapf.................................................(no files to check)Skipped
Check for added large files..............................................Passed
Check for merge conflicts................................................Passed
Check for broken symlinks................................................Passed
Detect Private Key...................................(no files to check)Skipped
Fix End of Files.....................................(no files to check)Skipped
clang-formater.......................................(no files to check)Skipped
[my-cool-stuff c703c041] add test file
 1 file changed, 0 insertions(+), 0 deletions(-)
 create mode 100644 233
```


## Keep the latest local repository

It needs to keep up with the latest code of original repository(<https://github.com/PaddlePaddle/Paddle>）before Pull Request.

Check the name of current remote repository with `git remote`.

```bash
➜  git remote
origin
➜  git remote -v
origin    https://github.com/USERNAME/Paddle (fetch)
origin    https://github.com/USERNAME/Paddle (push)
```

origin is the name of remote repository that we clone,which is also the Paddle under your own account. Next we create a remote host of an original Paddle and name it upstream.

```bash
➜  git remote add upstream https://github.com/PaddlePaddle/Paddle
➜  git remote
origin
upstream
```

Get the latest code of upstream and update current branch.

```bash
➜  git fetch upstream
➜  git pull upstream develop
```

## Push to remote repository

Push local modification to GitHub(https://github.com/USERNAME/Paddle).

```bash
# submit it to remote git the branch my-cool-stuff of origin
➜  git push origin my-cool-stuff
```
