# 关于android开发中的环境变量

在使用Eclipse的过程中有三个关于变量的问题我一直搞不懂,在坛子里询问高人.

1. 环境变量到底是添加java sdk的地址还是android sdk的地址.网上的教程很混乱,说什么的都有, 

2. 变量值已经存在怎么办,path已经被其他变量占用,该如何再添加其他变量? 

3. 我在安装java sdk后,将android sdk以及eclipse解压缩,启动eclipse,下载插件ADT以及制作简易helloworld都并没有什么错误出现,期间也并没有更改任何变量值.请问这样我是否还有必要添加变量,添加变量的主要作用是什么?

以下为答案:可以不添加环境变量，环境变量的作用是你在命令行模式下任意路径都可以直接使用android SDK Tools中的工具，而不用先把路径转到android SDK Tools中。

这句话里明白了几点,就是java sdk和android sdk的变量都是可以设置的,而设置了此变量后就可以不用再CD入sdk>tools,而之前Eclipse的找不到javaw.exe的错误也就消失了.

然后又有人解答了我其他的问题:在windows中设置环境变量,path值可以添加多个,只要用";"号隔开就可以了!

记得在以前我在[设置maya的环境变量](http://doo.hivan.net/archives/4.html)的时候也遇到了相应的问题,而那个时候我和[花儿](http://blog.istef.info/)进行过讨论.也是给出了符号隔开的问题.想不到现在又是同一个问题我就忘记以前的内容了....唉,年纪大了是不行了!

