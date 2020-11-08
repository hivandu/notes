# Mac MAX_open

在使用Hexo的过程里，经常会卡在deploy指令上，错误原因之一可能是因为Mac的MAX_open数小的原因，Linux默认为1024，而Mac上只有256，所以只要修改MAX_open数就可以了。指令如下：

<!--more-->

```
$ sudo sysctl -w kern.maxfiles=20480
kern.maxfiles: 12288 -> 20480

$ sudo sysctl -w kern.maxfilesperproc=18000
kern.maxfilesperproc: 10240 -> 18000

$ ulimit -S -n 2048
bubbyroom.com

$ ulimit -n
2048
```
其中，$ ulimit -n是用于查看Mac的MAX_open数的指令。只执行修改之前可以先执行此指令查看一下。

后记：在Terminal中修改了MAX_open仅适用于当前窗口，新建Tab，窗口后在新的Tab和窗口里都会失效。