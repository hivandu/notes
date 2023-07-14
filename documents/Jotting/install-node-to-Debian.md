# Debian上安装Node服务器

> 本教程只针对我自己，记录用，而且并不是完整版，不对其他人负责。请尽量不要参照本文


最近一直在学习NodeJS，本地上玩的差不多了，总要去架设个服务器跑一下，选择了**digitalocean**加的$5/mo 服务，安装了个Debian，至于为什么是Debian，好吧，因为。。。

<!-- more -->
---

言归正传，一遍操作一遍记录一下，好记性不如烂笔头嘛。

至于怎么注册DigitalOcean这里就不在详述了，从SSH登录开始吧。

## SSH登录

未免麻烦，最好是选择SSH登录，官方有详细的介绍：
[How To Use SSH Keys with DigitalOcean Droplets](https://www.digitalocean.com/community/tutorials/how-to-use-ssh-keys-with-digitalocean-droplets)

这里要说一下，DigitalOcean每次登录的时候都会告诉我密码过期，害得我重置了无数次密码。如果你也遇到这种问题，那么就先选择**Conole Access**, 然后在弹出的窗口控制台进行操作，修改root密码后再在本地操作。

```shell
// 创建SSH key
ssh-keygen -t rsa
Generating public/private rsa key pair.
Enter file in which to save the key (~/.ssh/id_rsa): 
Enter passphrase (empty for no passphrase): 
Enter same passphrase again: 
Your identification has been saved in ~/.ssh/id_rsa.
Your public key has been saved in ~/.ssh/id_rsa.pub.
The key fingerprint is:
4a:dd:0a:c6:35:4e:3f:ed:27:38:8c:74:44:4d:93:67 demo@a
The key's randomart image is:
+--[ RSA 2048]----+
|          .oo.   |
|         .  o.E  |
|        + .  o   |
|     . = = .     |
|      = S = .    |
|     o + = +     |
|      . o + o .  |
|           . o   |
|                 |
+-----------------+
```

```shell
cat ~/.ssh/id_rsa.pub
```
然后添加到digitalocean的SSH Keys里，Name随便起

之后我们就可以链接服务器了

```shell
cat ~/.ssh/id_rsa.pub | ssh root@[your.ip.address.here] "cat >> ~/.ssh/authorized_keys"
```
然后，就可以直接登录了:
```shell
ssh root@[your.ip.address.here]
```

## 安装Node

我选择的方式是源码安装

```shell
// update system
$ sudo apt-get update
$ sudo apt-get install git-core curl build-essential openssl libssl-dev

// Clone node
$ cd /usr/local/src
$ git clone https://github.com/nodejs/node
$ cd node

// select checkout
$ git tag
$ git checkout v4.4.7

// install
$ ./configure
$ make
$ sudo make install
```
漫长的等待，然后就可以查询了`$ node -v`， 这会就会出现安装的node version

## 安装NPM
```shell
$ wget https://npmjs.org/install.sh --no-check-certificate
$ chmod 777 install.sh
$ ./install.sh
$ npm -v
3.10.5
```
## 安装zsh(不是必要)
```shell
sudo apt-get zsh

git clone git://github.com/robbyrussell/oh-my-zsh.git ~/.oh-my-zsh
// copy defult zshrc
cp ~/.zshrc ~/.zshrc.bak
// set oh-my-zsh to use
cp ~/.oh-my-zsh/templates/zshrc.zsh-template ~/.zshrc
cash -s /bin/zsh
sudo shutdown -r now
```

## 安装Ruby
### 安装rbenv
```shell
git clone git://github.com/sstephenson/rbenv.git ~/.rbenv
# 用来编译安装 ruby
git clone git://github.com/sstephenson/ruby-build.git ~/.rbenv/plugins/ruby-build
# 用来管理 gemset, 可选, 因为有 bundler 也没什么必要
git clone git://github.com/jamis/rbenv-gemset.git  ~/.rbenv/plugins/rbenv-gemset
# 通过 gem 命令安装完 gem 后无需手动输入 rbenv rehash 命令, 推荐
git clone git://github.com/sstephenson/rbenv-gem-rehash.git ~/.rbenv/plugins/rbenv-gem-rehash
# 通过 rbenv update 命令来更新 rbenv 以及所有插件, 推荐
git clone git://github.com/rkh/rbenv-update.git ~/.rbenv/plugins/rbenv-update
# 使用 Ruby China 的镜像安装 Ruby, 国内用户推荐
git clone git://github.com/AndorChen/rbenv-china-mirror.git ~/.rbenv/plugins/rbenv-china-mirror
```

```shell
echo 'export PATH="$HOME/.rbenv/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(rbenv init -)"' >> ~/.bashrc

#
rbenv install 2.3.1
```

特么什么都能遇到，远端locale和本地不符，提示无法安装

```shell
sudo locale-gen en_US.UTF-8
```

**// or**

```shell
sudo dpkg-reconfigure locales
```

```shell
vim /etc/ssh/ssh_config
```
注释或删除`AcceptEnv LANG LC_*` (**服务器SSH配置**)
然后断开SSH重新登录，不行重启一下服务器，就好了。

```shell
sudo shutdown -r now
```

继续:
```shell
rebnv install 2.3.1
```


## 部署Nginx
```shell
$ sudo apt-get install nginx
```

其实Nginx也不是必要装的，Node自己可以跑服务！
```shell
nohup node app.js
```

但是如果要多域名的话，需要用到Nginx反代，额，这部分还不懂。再去研究下！

顺便，加一个删除Nginx的步骤:

```shell
sudo apt-get --purge remove nginx
sudo apt-get autoremove
dpkg --get-selections|grep nginx
// 罗列出与nginx相关的软件， 
nginx-common deinstall 
然后
sudo apt-get --purge remove nginx-common
```



