# Android 2.2 App2sd 问题

<p>其实比起以前版本的app2sd来说,设置是一样的!只是多了一个步骤,就是需要给rom添加一个sdext.然后所有的设置就和以前的版本一模一样了.</p>

<p>首先当然需要有一个已经分好区的sdcard,具体设置可以查看我以前的文,有ubuntu下进行分区的和windows下的!</p>

<p>然后需要最近版本的SPL和Radio,这个本人不提供了,可以自行解决!伸手党可以留下自己的邮箱,我提供下载地址!</p>

准备工作做足后,第一步就是需要在手机上建立一个sdext访问.这里提供一个文本文档:<a href="http://yunfile.com/file/hivandu/16325f3c/">下载： fr-patch134.zip</a> 

放在sdcard根目录,然后在连接手机的情况下在终端如下操作:

* `adb shell`
* `# sh /sdcard/fr-patch134.txt sdext`
* `busybox df -h`

<p>如果看到有/sd-ext分区,OK,以下的事情就顺理成章了,参考我以前发布的app2sd步骤操作就好了!</p>

<p>至此所有问题解决!</p>

<p>此处为后续更新,由于之前忽略了点东西,所以这里做一个补充!</p>

<p>由于2.2rom和以前版本的一些差别,在做app2sd之前,需要挂在system,sd-ext和data分区,这是需要注意的一点!挂在命令为:mount</p>

<p>例子: mount system</p>

<p>有什么不明白的再问吧!</p>
