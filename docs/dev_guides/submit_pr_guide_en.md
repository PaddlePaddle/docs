# Guide of submitting PR to Github

## Finish Pull Request create PR

Create an Issue to describe your problem and keep its number.

Switch to the branch you have created and click `New pull request`。

<img src="https://github.com/PaddlePaddle/FluidDoc/blob/release/1.1/doc/fluid/advanced_usage/development/contribute_to_paddle/img/new_pull_request.png?raw=true"  style="zoom:60%">


Switch to targeted branch:

<img src="https://github.com/PaddlePaddle/FluidDoc/blob/develop/doc/paddle/guides/08_contribution/img/change_base.png?raw=true">

A note of `resolve #Issue number` in PR description results in automatic close of corresponding Issue after the merge of PR.More details can be viewed [here](https://help.github.com/articles/closing-issues-via-commit-messages/)。

Then please wait for review.If there is any need to make a modification,you can update corresponding branch in origin following the steps above.

### Link Issue

If a PR created by solving a problem of an Issue needs to be associated with the Issue, refer to [Code Reivew promise](./code_review_en.html)

## Sign CLA and pass unit tests

### Sign CLA

For the first time to submit Pull Request,you need to sign CLA(Contributor License Agreement) to ensure merge of your code.Specific steps are listed as follows:

- Please check the Check in PR to find license/cla and click detail on the right to change into CLA website.

<div align="center">

<img src="https://github.com/PaddlePaddle/FluidDoc/blob/release/1.1/doc/fluid/advanced_usage/development/contribute_to_paddle/img/cla_unsigned.png?raw=true"  height="40" width="500">

 </div>

- Please click “Sign in with GitHub to agree” in CLA website.It will change into your Pull Request page after the click.

<div align="center">

<img src="https://github.com/PaddlePaddle/FluidDoc/blob/release/1.1/doc/fluid/advanced_usage/development/contribute_to_paddle/img/sign_cla.png?raw=true"  height="330" width="400">

 </div>


### Pass unit tests

Every new commit in your Pull Request will trigger CI unit tests,so please make sure that necessary comments have been included in your commit message.Please refer to [commit](local_dev_guide.html#permalink-8--commit-)

Please note the procedure of CI unit tests in your Pull Request which will be finished in several hours.

Green ticks after all tests means that your commit has passed all unit tests,you only need to focus on showing the Required tasks, the ones not showing may be the tasks we are testing.

Red cross after the tests means your commit hasn't passed certain unit test.Please click detail to view bug details and make a screenshot of bug,then add it as a comment in your Pull Request.Our stuff will help you check it.


## Delete remote branch

We can delete branches of remote repository in PR page after your PR is successfully merged into master repository.

<img src="https://github.com/PaddlePaddle/FluidDoc/blob/develop/doc/paddle/guides/08_contribution/img/delete_branch.png?raw=true">

We can also delete the branch of remote repository with `git push origin :the_branch_name`,such as:

```bash
➜  git push origin :my-cool-stuff
```

## Delete local branch

Finally,we delete local branch

```bash
# Switch to develop branch
➜  git checkout develop
# delete my-cool-stuff branch
➜  git branch -D my-cool-stuff
```

And now we finish a full process of code contribution
