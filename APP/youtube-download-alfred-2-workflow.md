---
title: Youtube Download Alfred 2 Workflow
date: 2013-8-28 14:31:23
categories:
- develop
tags:
- Alfred
- workflow
---

在Youtube上浏览某个视频的时候，有没有很想下下来的冲动？毕竟有的时候VPN和SSH都不是太给力的时候，Youtube并不流畅，是慢慢加载？反正我是喜欢下载到本地。

<!--more-->

Chrome有很多下载Utube视频的插件，而我今天要介绍的，是Alfred的workflow -> Youtube Download Alfred 2 Workflow。

![Youtube Download Alfred 2 Wordflow](http://farm4.staticflickr.com/3831/9613403353_04b6f68151_z.jpg)

在使用这个workflow的之前，我们还需要做一些工作，当然，设置完毕之后是很爽的，so，之前的麻烦都是值得的，不嫌麻烦的就跟着我一起做吧。

首先，我们需要下载[youtube-dl](http://rg3.github.io/youtube-dl/),我们需要wget命令来获取它，好吧，比较麻烦的是，OSX默认是没有wget命令的，So,让我们再绕一圈吧，先去安装wget.

```
$ curl -O http://ftp.gnu.org/gnu/wget/wget-1.13.tar.gz
$ tar -xzvf wget-1.13.tar.gz
$ cd wget-1.13
$ ./configure --with-ssl=openssl
$ make
$ sudo make install
```

测试一个wget命令

```
$ wget --help
```

OK，最后让我们删除下载的文件。

```
$ cd .. && rm -rf wget-*
```

然后，我们可以获取youtube-dl了。

```
$ sudo wget https://yt-dl.org/downloads/2013.08.28/youtube-dl -O /usr/local/bin/youtube-dl
$ sudo chmod a+x /usr/local/bin/youtube-dl
```

之后，去下载我们今天的主角,[Youtube Download Alfred 2 Workflow](http://dferg.us/youtube-download-alfred-2-workflow/).

好了，尽情享用吧！