# VPS 设置 Hexo

首先需要感谢@[lucifr](http://lucifr.com/)，我现在这篇文就是在iPad上登录VPS完成的。最后还是忍不住入手了下边两个APP:

![Prompt](http://qiniu.hivan.me/blog/img/snap_prompt.jpg)

<!-- more -->

当然我其实到现在并不完美，因为rsync和自动执行`generate`的代码我没有完成。安装[incrond](http://inotify.aiken.cz/?section=incron&page=download&lang=en)的时候总会出错，于是无法执行集群文件同步.所以现在还是在终端里执行`generate`和`cp -rf /home/xxxx/* /home/xxxx`

我这里并不是要教设置步骤，因为其实@lucifr 已经在他的[这篇文](http://lucifr.com/2013/06/02/hexo-on-cloud-with-dropbox-and-vps/)里写的很清楚了，我就写几点注意事项

1. 搞定VPS操作和基本的Linux命令很重要。
2. 要搞定lnmp，参照这里的lnmp详细介绍
3. 新版本的Hexo有更改，在同一目录里是找不到/cli/generate.js的，更别说console.log语句了
4. @lucifr所说的新建立一个Dropbox账户，意思是在VPS主机上建立一个账户用来执行Dropbox同步，而不是新建立一个Dropbox账户。
5. 其他…

好吧，写其他是因为iPad上用VI进行编辑实在有点难受，现在先这样了，以后有时间了再写一个更详细的。

![Prompt](http://qiniu.hivan.me/blog/img/snap_prompt_ssh.jpg)