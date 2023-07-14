# G1 app2sd 完全教程

<p>声明1:你所需要的软件在<a href="http://hivan.me/2009/04/23/g1%E5%88%B7%E6%9C%BA%E9%A1%BA%E5%BA%8F%E6%8C%87%E5%8D%97.html">这里可以下载</a>的到!<br />
声明2.app2sd虽然可以省却手机内存,但是也有许多不便的地方!操作后SDcard就是机子的一部分,不能随便摘取.我用的4G的卡,在机子挂在后存储有问题!不知道其他卡如何.所以在存储文件和音乐的时候还是需要用到读卡器,而这个时候我必须选择关机!直接卸载SDcard会造成机子程序出错!而不得不从新执行一遍app2sd的过程!并且执行过后也会存在一些不可知的问题!如果对稳定性比较看重的人这里可以飘过了!<span style="color: #000000;">
<del datetime="2009-04-24T08:21:41+00:00">声明3.我的sdcard已经在手机内通过!懒得再刷,所以没有用我的card抓图!本教程图片多为网上现成图片来完成!而图片不是一个地方抓取的!所以图片上的容量会有差距.但是刷机过程没有错</del></span></p>

<p><span style="color: #ff0000;">从新格盘,正好用自己的图!顺便说一下,ubuntu下的默认抓图真恶心!每抓一张都要从新启动一次程序!</span></p>

<p><!--more-->
所需要的准备的工作:<br />
1.SDcard(必须)<br />
2.分区软件(必须,windows下可以使用Acronis Disk Director Suite,支持vista.linux下可以直接利用终端分区!)<br />
3.Android SDK(非必须,可以再网上下载Terminal Emulator.apk,安装后在手机上输入adb下的指令完成操作!)<br />
首先我们要将SDcard分区,分成fat32和ext2,至于ext3是否可行我没有测试过,有兴趣的可以试试并且留言告诉我测试报告!</p>

<p>我选择的是在ubuntu的终端执行,这样操作比较靠谱.而在windows下的分区软件不是很稳定!会造成诸多不可见的错误!</p>

<p>windows下的分区软件有Acronis Disk Director Suite以及PartitionManager,至于分区魔术师可以略过,因为它不支持分区SDcard.Acronis Disk Director Suite软件分区可以<a href="http://www.whatgp.com/thread-411-1-1.html">移步到此查看</a>!</p>

<p>以USB内存卡方式插上电脑，或者用读卡器插上电脑<br />
像我的ubuntu，它会自动挂载你的卡。
<strong>把东西备份好，然后卸载。一定要卸载，不然无法分区</strong>
<img src="http://farm4.static.flickr.com/3528/3469758169_cc4947e02e.jpg?v=0" alt="" />
启动ubuntu或者您的linux系统,在终端内输入如下代码:
<blockquote>dmesg   //查看所连接的设备!</blockquote>
<img src="http://farm4.static.flickr.com/3590/3469737465_27a1c78941.jpg?v=0" alt="" />
可以看到sdb或者sdc之类的设备名称!假设我以下操作都为sdc设备!
<blockquote>sudo fdisk /dev/sdc   //这里需要说明,如果linux下非root,必须要输入sudo来取得root权限进行操作.以下类同!</blockquote>
p是显示当前分区<br />
n是创建<br />
d是删除<br />
w是应用你的操作
<blockquote>doo@ubuntu:~# sudo fdisk /dev/sdc<br />
Command (m for help): d &lt; ==删除当前分区<br />
Command (m for help): p &lt;==显示一下，确定已删除<br />
Disk /dev/sdc: 3965 MB, 3965714432 bytes<br />
122 heads, 62 sectors/track, 1024 cylinders<br />
Units = cylinders of 7564 * 512 = 3872768 bytes<br />
Disk identifier: 0x9dfd42a5</blockquote></p>

<p>Device Boot      Start      End      Blocks    Id    System</p>

<p>Command (m for help):
<img src="http://farm4.static.flickr.com/3593/3470552218_eea0800e72.jpg?v=0" alt="" />
<blockquote>Command (m for help): m &lt; ==查看帮助<br />
Command action<br />
a   toggle a bootable flag<br />
b   edit bsd disklabel<br />
c   toggle the dos compatibility flag<br />
d   delete a partition<br />
l   list known partition types<br />
m   print this menu<br />
n   add a new partition<br />
o   create a new empty DOS partition table<br />
p   print the partition table<br />
q   quit without saving changes<br />
s   create a new empty Sun disklabel<br />
t   change a partition's system id<br />
u   change display/entry units<br />
v   verify the partition table<br />
w   write table to disk and exit<br />
x   extra functionality (experts only)<br />
Command (m for help): n &lt;==新建分区，选择主分区<br />
Command action<br />
e   extended<br />
p   primary partition (1-4)<br />
p
Partition number (1-4): 1 &lt;==指定该主分区为1号<br />
First cylinder (1-1024, default 1): &lt;==敲回车，直接使用SD卡的最开头<br />
Using default value 1<br />
Last cylinder or +cylinders or +sizeK(K,M,G) (1-1024, default 1024): +3300M &lt;==填入分区的大小<br />
Command (m for help): n &lt;==新建分区，选择扩展分区(所有逻辑分区加起来就是扩展分区)</blockquote></p>

<p>Command action<br />
e   extended<br />
p   primary partition (1-4)<br />
p
Partition number (1-4): 2 &lt;==扩展分区的序号是2<br />
First cylinder (895-1024, default 895):   &lt;==敲回车，直接接着剩余空间的最开头<br />
Using default value 895<br />
Last cylinder or +cylinders or +sizeK(K,M,G) (895-1024, default 1024):  &lt;==敲回车，用默认的，使用全部剩余空间<br />
Using default value 1024<br />
Command (m for help):
<img src="http://farm4.static.flickr.com/3553/3470552294_bb091b9291.jpg?v=0" alt="" />
<blockquote>Command (m for help):p</blockquote></p>

<p>Disk /dev/sdc: 3965 MB, 3965714432 bytes<br />
122 heads, 62 sectors/track, 1024 cylinders<br />
Units = cylinders of 7564 * 512 = 3872768 bytes<br />
Device Boot      Start         End      Blocks   Id  System<br />
/dev/sdc1               1           894      733792+  83  Linux<br />
/dev/sdc2             729         1024      272128+  83  Linux
<strong>创建好两个分区后, 我们还需要用命令t修改分区卷标, 选择分区1改卷标为c</strong>
命令为
<blockquote>Command (m for help):t t &lt; ==修改卷标<br />
partition number (1-4): 1 &lt;==输入1来制定第一个分区.<br />
Hex code (type L to List codes): c &lt;==输入C来制定卷标<br />
Changed system type of partition 1 to c (W95 FAT32 (LBA))</blockquote></p>

<p>Command (m for help): w &lt;==将缓冲写入SD卡,应用你的操作</p>

<p>The partition table has been altered!<br />
Calling ioctl() to re-read partition table.</p>

<p>WARNING: If you have created or modified any DOS 6.X<br />
partitions, please see the fdisk manual page for additional<br />
information.<br />
Syncing disks.<br />
doo@ubuntu:~#
<img src="http://farm4.static.flickr.com/3532/3470552494_7fe52d0073.jpg?v=0" alt="" />
<blockquote>doo@ubuntu:~# sudo ls /dev/sdc* &lt; ==查看分区情况<br />
/dev/sdc  /dev/sdc1  /dev/sdc2<br />
doo@ubuntu:~#sudo mkfs.vfat /dev/sdc1 &lt;==格式化第一个主分区。<br />
mkfs.vfat 3.0.1 (23 Nov 2008)<br />
doo@ubuntu:~# sudo mkfs.ext2 /dev/sdc2 &lt;==格式化第二个分区<br />
mke2fs 1.41.4 (27-Jan-2009)<br />
warning: 139 blocks unused</blockquote></p>

<p>Filesystem laber=<br />
OS type: Linux<br />
Block size=1024 (log=0)<br />
Fragment size=1024 (log=0)<br />
123360 inodes, 491521 blocks<br />
24583 blocks (5.00%) reserved for the super user<br />
First data block=1<br />
Maximum filesyetem blocks=67633152<br />
68 block groups<br />
8192 blocks per group, 8192 fragments per group<br />
2856 inodes per group<br />
Superblock backups stored on blocks:<br />
8193, 24577, 40961, 57345, 73729, 204801, 221185, 401409</p>

<p>Writing inode tables: done<br />
Writing superblocks and filesystem accounting information: done</p>

<p>This filesystem will  be automatically checked every 38 mounts or 180 days.whichever comes first. Use tune2fs -c or -i to override.<br />
doo@ubuntu:-$
<img src="http://farm4.static.flickr.com/3623/3470552584_4e666b7a7e.jpg?v=0" alt="" />
分区完毕后ubuntu会自动挂在两个盘符.表示成功!
<img src="http://farm4.static.flickr.com/3500/3469757943_3819d59449.jpg?v=0" alt="" /></p>

<p>然后需要手机必须为app2sd版本的rom,在windows 命令提示符下输入命令查看:<br />
以下步骤必须安装android sdk.(其实一下步骤不一定需要在windows cmd下进行,在网上下载一个android的终端Terminal Emulator.apk,然后启动此程序在手机内输入以下指令是一样的!只是在sdcard的系统盘下建立app文件夹并挂载到android rom上! )
<blockquote>C:\Documents and Settings\doo&gt;cd c:\sdk\tools &lt; ==cd到sdk adb.exe<br />
C:\sdk\tools&gt;adb devices &lt; ==查看连接的硬件和设备<br />
List of devices attached<br />
000000000000    device &lt;==分区过硬盘以后连接会显示000000000000 的硬件号</blockquote></p>

<p>C:\sdk\tools&gt;adb shell<br />
# su &lt; ==如果你还没有取得root权限,那么这一步通不过.<br />
su<br />
# ls /system  &lt;==查看一下system目录下的文件夹<br />
ls /system<br />
lib<br />
framework<br />
media<br />
fonts<br />
etc<br />
customize<br />
build.prop<br />
usr<br />
bin<br />
xbin<br />
app<br />
sd<br />
lost+found
<img src="http://farm4.static.flickr.com/3593/3469841621_c654117ee4.jpg?v=0" alt="" />
<blockquote># busybox df -h  &lt; ==查看系统盘情况!如果分区成功,那么会在android的系统下显示分区.如下我的385.8M的分区在android的系统内!再往下是sdcard的系统!如果没有那表示分区失败.<strong>当然还有一种可能就是你的手机不是app2sd rom<br />
busybox df -h<br />
Filesystem                Size      Used Available Use% Mounted on<br />
tmpfs                    48.3M         0     48.3M   0% /dev<br />
tmpfs                     4.0M     12.0k      4.0M   0% /sqlite_stmt_journals<br />
/dev/block/mtdblock3     67.5M     67.5M         0 100% /system<br />
/dev/block/mtdblock5     74.8M     30.4M     44.3M  41% /data<br />
/dev/block/mtdblock4     67.5M      1.2M     66.3M   2% /cache<br />
/dev/block/mmcblk0p2    385.8M      2.0k    366.5M   0% /system/sd  &lt; ==由于在ubuntu下分区后手机内读取sdcard出错,所以后便又分了一次!但是没有抓图,所以容量上和上图有差距.再者本身linux和windows读取SDcard的容量上就有不同!<br />
/dev/block//vold/179:1<br />
3.3G      4.0k      3.3G   0% /sdcard<br />
# mkdir /system/sd/app &lt;==建立sdcard分区上的app文件夹!如果以前sdcard曾做过app2sd,那么这个文件夹是存在的!会有命令符提示文件夹存在!<br />
mkdir /system/sd/app<br />
# cd /data<br />
cd /data<br />
# cp -a app /system/sd/app<br />
cp -a app /system/sd/app<br />
# rm -r app<br />
rm -r app<br />
# ln -s /system/sd/app /data/app<br />
ln -s /system/sd/app /data/app<br />
# reboot<br />
reboot</strong></blockquote>
<strong><img src="http://farm4.static.flickr.com/3502/3469841579_4f7fa48f60.jpg?v=0" alt="" />
手机自动重启后就OK了.放心安装你所想要的apk程序吧!
</strong><strong>
顺便说一句:ubuntu的9.04快要放出正式版了!欢迎大家下载试用.</strong></p>
