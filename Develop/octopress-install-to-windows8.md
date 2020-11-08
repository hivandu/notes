---
title: Windows 8安装Octopress记录
date: 2012-4-7 11:59:25
categories:
- develop
tags:
- windows
- octopress
- code
---

### 前言

虽然昨晚因为代码高亮问题有想过放弃([见此文](http://hivan.me/2012-w16-06/)),但今早上网查阅的时候忽然发现一篇自己以前从来没Google到的[文章](http://blog.sprabbit.com/blog/2012/03/23/octopress/),所以抱着试试的心态又尝试了一下!果然解决了.其实说起来搞笑,那么多技术性文章,原来最终的原因简单的知识因为我安装的是64bit的Python,而ruby对windows下64bit的Python支持非常差! **安装32bit Python后问题如愿解决**.下边做一个这几天的完整记录.

### 安装Ruby

<!-- more -->

Octopress是基于Ruby的项目,然后我们首先需要Ruby环境才可以跑,在Linux下,是使用[RVM](http://www.ruby-lang.org/en/downloads)进行安装编译,但是Windows下,我们有两个选择,一个就是[RubyInstaller](http://rubyforge.org/frs/?group_id=167) + [Development Kit](https://github.com/oneclick/rubyinstaller/wiki/development-kit),再有就是直接安装[RailsInstaller](file:///Users/Hivan/Downloads/hexopress-gh-pages/octopress-install-to-windows8/railsinstaller.org/),所有的Octopress所需环境一次性安装完毕!

需要用到DevKit的原因是因为安装Octopress的时候,需要用到的Ruby gems会需要在本地编译(Ex: [rdiscount](https://github.com/rtomayko/rdiscount) ),而Devkit这是一套基于[MSYS/MinGW](http://www.mingw.org/wiki/MSYS)下的C/C++编译环境工具组.而RailsInstaller则自带Devkit.所以使用RailsInstaller这是一劳永逸的方法!

部署好这两个工具后,首先需要以下指令(在CMD中):

```
cd c:\Devkit #你自己的Devkit目录
ruby dk.rb init #用来产生config.yaml,里边会有你的ruby路径,一般会帮你设定好!
ruby dk.rb install
```

之后,我们需要更新一下gem,也是为了保险起见(在CMD中)!

```
gem sources -l  #查看源地址
gem sources -a http://ruby.taobao.org/ #添加淘宝的ruby源地址
gem sources --remove http://rubygems.org/ #删除rubygems源
gem sources -l #再查看一下源地址,确保只有http://ruby.taobao.org/一个源
gem update --system
gem update
gem install rdoc bundler
```

至此,Ruby环境部署完毕!

### 安装Git

要使用Github部署自己的博客,首先我们需要安装Git,下载[msysgit](http://code.google.com/p/msysgit/downloads/list)之后直接安装就OK.

### 安装Octopress

启动你的Git Bash,新建一个目录用来下载Octopress `$ mkdir new_repos`

然后进入你的目录下载和部署你的Octopress

```
$ cd new_repos
$ git clone git://github.com/imathis/octopress.git DIR_NAME #这里是你的Octopress实际保存目录
$ cd DIR_NAME
$ notepad Gemfile #修改你的ruby索取地址,将"http://rubygems.org/"修改成"http://ruby.taobao.org/"
$ gem install bundler
$ bundle install #安装Gemfile档案中所列的gems,也是Octopress所需Gem组件
$ rake install #安装预设的Octopress theme
```

有可能这个时候rake命令会出现错误!如:

```
rake aborted!
You have already activated rake 0.9.2.2, but your Gemfile requires rake 0.9.2.
Using bundle exec may solve this.
(See full trace by running task with --trace)
```

在这里需要调整一下rake

```
$ echo "alias rake='bundle exec rake'" >> ~/.bash_profile
bundle update
```

然后在`rake install`生成默认模板就没问题了!

如果只是写一般的文字日志,那么到这里本地部署就基本结束!可以部署Github和本地仓库的关联.可以跳过以下的Python环境设置步骤,如果你还想要分享一些代码,也可以使用markdown内附的code block syntax,但是你希望可以代码高亮!那么就必须安装python 2.7,注意,是2.7版本的!并不是越高版本越好,因为我们高亮是需要一个Python的开源项目Pygment,虽然Pygment支援 Python 2 版本和3版本,但是Ruby和Python之间的桥接是RubyPython完成,而RubyPython目前只支援Python 2!

### 部署Python

我们首先要下载Python安装,在这里推荐[ActivePython 2.7](http://www.activestate.com/activepython) ,这里注意,一定要下载32bit的版本! 别管你系统是否64bit的,原因如我一开始所说!

安装完毕后无需你自己设置环境变量,安装的时候会自己加上变量到你机子的path中!

其实这里是一个难点!因为很多问题都会出现在这里.我也是在这里卡了好几天!基本所有时间都耗费在这一步.就为了一个代码高亮,和它死磕了近一个星期.在这期间遇到了各种各样的问题.好吧,如果你安装了32bit的Python在渲染高亮的时候还是遇到了问题,那么你可以做如下尝试:

- Could not open library'.dll': The specified module could not be found

查看错误消息是在执行rubypython.rb中的函数时产生了错误.无门需要修改其中的一些代码: 对ruby目录下的`lib\ruby\gems\ruby 1.9.x>\gems\rubypython-0.5.x\lib\rubypython\pythonexec.rb`做[如下修改](https://github.com/bendoerr/rubypython/commit/1349aea1c6faa459c4be8474e4a7e878f08459c2).

有可能我们需要将System32下的Python27.dll以及其他相关dll文件copy到sysWOW64下和python /libs目录下! 如果还不行,请使用一个黑暗的解决方法: 修改你的文件:`lib\ruby\gems\1.9.1\gems\rubypython-0.5.3\lib\rubypython\python.rb`

`<notextile>0</notextile>`

- Liquid error: undefined method `Py_IsInitialized’ for RubyPython::Python:Module

这是我一直遇到的问题!因为之前安装ActivePython的时候程序已经将路径加入到变量下,所以只能参考这个issue和这个issue来解决问题.

- 尝试安装[Python-devel](http://sourceforge.net/projects/pywin32/)

- 尝试修改`lib\ruby\gems\1.9.1\gems\rubypython-0.5.3\lib\rubypython\pythonexec.rb`

pythonexec.rb
```
    def initialize(python_executable)
...
    @realname = @python.dup
    if (@realname !~ /#{@version}$/ and @realname !~ /\.exe$/)
    @realname = "#{@python}#{@version}"
``` 

修改成:

`<notextile>2</notextile>`

### UTF-8 编码

Windows预设是Big5编码,所以要想'rake generate'的时候不报编码错误,我们需要设置一下编码!方法有两个,一直是直接在Git中设置变数:

```set LANG=zh_CN.UTF-8
set LC_ALL=zh_CN.UTF-8
```

还有一个是在变量中加入这两个变量: 新添加`LANG`和`LC_ALL`,参数为`zh_CN.UTF-8`.然后在Git中做如下设置: $ echo "export LANG LC_ALL" > ~/.bash_profile

至此,本地环境就结束了! 你可以尝试如下命令来查看自己的Octopress. rake new_post['postname'] #新建post rake generate #生成HTML rake preview #在本地http://localhost:4000 中查看!

### 建立远程Git仓库链接

这部分我一直还没弄明白,关于分支等等问题都还搞不明白! 对Git熟悉的朋友可以跟我留言指导!

首先我们需要生成一个`ssh public key`,然后添加到新建的github帐户中!

关于这部分,因为我也并不熟悉,不好瞎指导,所以可以参看Github的[官方帮助文档](http://help.github.com/)

添加完毕后我们就需要建立仓库链接:

```
$ cd RID_NAME #你的Octopress目录
$ rake setup_github_pages #和Github建立连接
```

然后根据提示输入github URL,格式为: `git@github.com:your_username/your_username.github.com.git`

然后直接 `$ rake deploy`就将本地的文件上传到远程仓库了! 这里请注意以下,要看你在_config.yml中的设置.有可能是将public/name'这个目录下的文件cp到_deploy目录后上传,也有可能就是public`目录下.

以后在更新博客就直接如下命令: `$ rake new_post['newname']` #建新文章 $ rake generate #更新HTML文件 $ rake push #更新远程仓库 关于Octopress的设置问题,基本都在_config.yml文件中,这部分我自己也不太清楚,也就不写了!所以,本文到此结束!

在这里,感谢以下文档的作者,这期间这些文档帮了不少忙!还有,伟大的Google:

- [Octopress官方文档](http://octopress.org/docs/)
- [github官方文档](http://help.github.com/)
- [關於在64位 Windows 7 中部署中文化的Octopress](http://blog.sprabbit.com/blog/2012/03/23/octopress/)
- [在 Windows7 下从头开始安装部署 Octopress](http://sinosmond.github.com/blog/2012/03/12/install-and-deploy-octopress-to-github-on-windows7-from-scratch/)
- [试用Octopress](http://www.blogjava.net/lishunli/archive/2012/03/18/372115.html)
- [在 Windows 使用 Octopress - 不歸錄](http://tonytonyjan.heroku.com/2012/03/01/install-octopress-on-windows/)

PS: 最后,给大家一个建议! 浏览器上安装一个Evernote插件,用来保存自己Google到的文档!便于以后查阅和参考! :) 预祝大家顺利...

