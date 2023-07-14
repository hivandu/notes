# 模拟器的app2sd

一个读者**@XXX(因为个人意愿隐掉名字)** 发mail询问我关于windows下模拟器app2sd的问题,先不说有没有必要,说一下我测试的结果! 本来我给他回邮件是让他试一下apptosd.apk的,后来越想越不对劲,所以自己做了下测试!结果如下:

![](http://farm3.static.flickr.com/2654/3806259231_f347d9ba50.jpg)

然后通过adb shell也无法操作. # mkdir /system/sd/app
mkdir /system/sd/app
mkdir failed for /system/sd/app, No such file or directory
不过仔细想想,app2sd必须满足的条件我们在windows上根本就不存在,首先我们必须要一个app2sd的支援固件!然后我们需要sdcard分出一个ext2的分区...
而这两个条件全部都不满足!那基本上可以说没有办法!

看以后有没有高手可以实现模拟器上安装修改固件,那么可以安装一个app2sd的固件,而另外一个必须满足的条件就是必须将建立的虚拟sdcard分出一个ext2分区来!

满足了这两个条件,那么所有的都会水到渠成!

BTW:下午这位读者给我的回复:

非常感谢！
我测试的结果跟您是一样的，不过后面那个建目录的不一样。下面是我对您博文上的一点分析。 然后通过adb shell也无法操作. 【XXX】google好象改过linux内核，adb shell登录以后几种命令都有权限限制。在虚拟sd卡上建立文件夹有所有权限，然而用adb push传上去的就没有可执行的权限，使用chmod命令修改权限也不成功。 `# mkdir /system/sd/app`

```
mkdir /system/sd/app
mkdir failed for /system/sd/app, No such file or directory 
```

【XXX】在system目录下adb shell 命令是没有写权限的。你这个尝试如果是mkdir /system/sd就会报“mkdir failed for sd, Read-only file system”的错误，但是在data目录下就能够创建目录。 不过仔细想想,app2sd必须满足的条件我们在windows上根本就不存在,首先我们必须要一个app2sd的支援固件!然后我们需要sdcard分出一个ext2的分区…
而这两个条件全部都不满足!那基本上可以说没有办法!
看以后有没有高手可以实现模拟器上安装修改固件,那么可以安装一个app2sd的固件,而另外一个必须满足的条件就是必须将建立的虚拟sdcard分出一个ext2分区来! 【XXX】appsd的固件这个是什么概念？用mksdcard创建的虚拟sdcard不就是对应的手机上的sdcard么？虚拟sdcard为什么要分出一个ext2分区呢？ext2分区一个什么概念，sdcard要分出ext2分区的原理是什么？
能否简单介绍一下？或者介绍一下相关的资料？谢谢。
另：对您给我传的那个apk我不是很了解，这个文件从哪里来的，它都做了些什么事？
下面是我adb shell后ls –l查看到的各文件夹权限，是有加载虚拟sdcard的。

```
drwxrwxrwt root root 2009-08-10 04:45 sqlite_stmt_journals
drwxrwx--- system cache 2009-07-21 09:01 cache
d---rwxrwx system system 2009-08-10 04:52 sdcard
lrwxrwxrwx root root 2009-08-10 04:45 etc -> /system/etc
drwxr-xr-x root root 2009-05-15 00:53 system
drwxr-xr-x root root 1970-01-01 00:00 sys
drwxr-x--- root root 1970-01-01 00:00 sbin
dr-xr-xr-x root root 1970-01-01 00:00 proc
-rwxr-x--- root root 9075 1970-01-01 00:00 init.rc
-rwxr-x--- root root 1677 1970-01-01 00:00 init.goldfish.rc
-rwxr-x--- root root 106568 1970-01-01 00:00 init
-rw-r--r-- root root 118 1970-01-01 00:00 default.prop
drwxrwx--x system system 2009-05-15 00:58 data
drwx------ root root 1970-01-01 00:00 root
drwxr-xr-x root root 2009-08-10 04:46 dev
```

这里请注意system，sdcard和data它们的权限以及各自的意义。

```
d---rwxrwx system system 2009-08-10 04:52 sdcard
drwxr-xr-x root root 2009-05-15 00:53 system
drwxrwx--x system system 2009-05-15 00:58 data
```

下面说下我对各个信息的理解。首先第一列这是表示的各用户的权限，d代表这是文件夹，rwx分别代表读、写、执行权限。
d后面第一组三个字符表示当前用户的读写执行权限，第二组代表group用户的权限，第三组表示other用户的权限。
然后是第二列，表示当前用户对该文件夹的权限级别，第三列代表该文件夹的当前用户。
如果我对这组信息的含义理解方式正确的话，那么这里我就有疑问了：
1. linux下面有system这个权限级别吗？我有个同事说只有root、group和other，所以我很奇怪这里的system这个权限级别是怎么回事，它有什么样的权限，能做到怎么样。
2. sdcard这个目录非常奇怪，自己的用户权限都没有，group和other用户却有所有权限，在sdcard目录里面建立的目录权限跟sdcard的权限一样。
3. 我们自己写的应用程序，不知道是属于什么样的权限级别，是作为什么样的用户来访问各目录包括sd卡的，手机sd卡和虚拟sd卡。

