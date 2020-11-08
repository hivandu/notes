# Gphone SD卡分区(更新)

<p>以前写过一个<a href="http://doo.hivan.net/archives/877.html">Gphone app2sd的完全教程</a>,在那个教程当中有完整的分区步骤,可是那个分区模式是需要linux系统才可以执行,而我当时用的是ubuntu.</p>

<p>现在可行的分区模式也就是linux下,然后windows下的分区软件.譬如PartitionManager,AcronisDiskDirector,不要去想PQ魔术师,那个不支持sd卡分区!就这么多了么?...其实,如果你有Gphone手机的话,完全可以用windows dos来分区!
<!--more-->
先说条件:<br />
1.Gphone手机<br />
2.SDcard<br />
3.android sdk<br />
4.一条usb连接线</p>

<p>步骤:<br />
首先开机进入recovery模式，按ALT+X进入“console”</p>

<p>打开cmd,输入:
<code>adb shell<br />
parted /dev/block/mmcblk0<br />
print</code>
可以看到分区的情况,一般来说都是一个分区,如果以前做过app2sd那么就是两个...删除这两个分区.如下图为三个分区:
<img src="http://farm4.static.flickr.com/3525/3883103722_94329d34bb.jpg" alt="" />
然后输入代码删除分区:
<code>rm 1<br />
rm 2<br />
rm 3</code>
(如果只有两个分区或者一个分区的,执行一步操作就可以了,也就是rm< 数字>)<br />
在完成后就是一个未分区的SDcard
<img src="http://farm4.static.flickr.com/3040/3883103786_36c390b7e0.jpg" alt="" />
然后对SDcard重新分区,注意需要根据你的卡大小来分配各分区的大小,一般linux-swap最大32M.ext分区500M足够了,最大不要超过1GB.不过似乎有将linux-swap分成96Mb的...输入以下代码分区:
<code>mkpartfs primary fat32 0 7445<br />
mkpartfs primary ext2 7445 7873         300-500都可<br />
mkpartfs primary linux-swap 7873 7969      务必是96M 不然你有C6卡也不能全速体验HERO了...  </code>
至此,分区工作就完成了.可以输入print来检查一下.<br />
以下为可执行可不执行步骤...就是将ext2转换为ext3/ext4<br />
转换为ext3输入以下命令:
<code>upgrade_fs</code>
转换ext4输入以下命令:
<code>tune2fs -O extents,uninit_bg,dir_index /dev/block/mmcblk0p2<br />
e2fsck -fpDC0 /dev/block/mmcblk0p2<br />
upgrade_fs </code>
结束以后,输入
<code>parted /dev/block/mmcblk0<br />
print</code>
验证是否升级到ext3/ext4<br />
然后quit退出,重启手机.</p>

<p>本教程参考了<a href="http://www.hiapk.com/bbs/thread-17147-1-2.html">安卓网上安装hero的步骤</a></p>
