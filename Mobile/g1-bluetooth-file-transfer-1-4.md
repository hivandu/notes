# G1 蓝牙传输更新 Bluetooth File Transfer 1.4

<p>不知道有没有人和我遇到了一样的问题!在G1rom更新到固件1.6以后,app2sd的Bluex1.12版本无法安装,而无app2sd版本的在安装后只能发送文件而无法打开蓝牙端口接收!</p>

<p>有兴趣的朋友可以<a href="http://doo.hivan.net/archives/967.html">到这里下载</a>以后试试.
<!--more-->
而这次寻到一个新的软件,版本为1.4版本,相信不是以前的bluex的版本更新,其实我自己也不确定是否为更新版本,因为图标已经换的很彻底了,并且内部结构也与以前大不一样!最重要的一点,原来支持文档类别存放,而现在却统一放在一个文件夹下!确实有些很不方便...</p>

<p>不过这个apk只需要在sdcard内直接安装就可以使用,需要提供root权限...而传输和接受都OK,并且比较顺畅!</p>

<p>另外,此软件我没有在hero的固件上测试,不知道是否支持hero,有兴趣的自己试试吧!而hero上的蓝牙传输其实有解决办法了...如下:<br />
*More progress 11:04pm CST 8/26/09*<br />
Tracked down what calls the BTIP service, it's /system/lib/libandroid_runtime.so . Tried replacing it with a cupcake build, rebooted and ran into the issue where /system/framework/framework.jar is still referencing calls that were in the Hero libandroid_runtime.so . So replaced framework.jar and framework.odex from cupcake build and got the following error.
<code>D/AndroidRuntime( 1517): >>>>>>>>>>>>>> AndroidRuntime START < <<<<<<<<<<<<<<br />
D/AndroidRuntime( 1517): CheckJNI is OFF<br />
I/dalvikvm( 1517): DexOpt: mismatch dep signature for '/system/framework/core.odex'<br />
E/dalvikvm( 1517): /system/framework/framework.jar odex has stale dependencies<br />
I/dalvikvm( 1517): Zip is good, but no classes.dex inside, and no valid .odex file in the same directory<br />
D/libc-abort( 1517): abort() called in pid 1517</code>
Any "educated" ideas?</p>

<p>*Questions & Progress 01:09pm CST 8/17/09*<br />
So lately what I've been trying to do is find where a reference is made to actually call the BTIPS service. I've been lookiing in /system/framework and /data/app_s/Settings.apk but haven't found it yet. What I'm hoping to do is modify the file and have it call BT the same way cupcake did. Has anyone else found where a reference to "btips" is at?</p>

<p>Settings.apk, which is what pops up when on home screen and you hit menu->settings, only makes a call to "android:targetClass="com.android.settings.bluetoo th.BluetoothSettings"</p>

<p>Anywho, if you find it in any system libraries or framework files let me know. Please no PM's or posts about where you "THINK" it may be at. I've already tried the random guessing stuff, now I'm going through libraries one by one trying to find it.</p>

<p>*Some more notes 12:30pm CST 7/24/09*<br />
Here are some notes of interest.</p>

<p>There are two versions of the /system/bin/bts daemon that are floating around on the Hero builds
<code>md5sum bts<br />
29ffa46f12c01e3690690752b4e2d58d  bts</code></p>

<p>md5sum bts<br />
5aeaca42d67d3b3c64ceda9ee4bfec1a  bts
There are also two versions of the TIInit_5.3.53.bts firmware files. One is actually just the brf6300.bin file renamed to match what Hero is looking for in /etc/firmware
<code>md5sum TIInit_5.3.53.bts<br />
d7a214bdb9b4fbc2b4e2dd7e3ab95df0  TIInit_5.3.53.bts</code></p>

<p>md5sum TIInit_5.3.53.bts<br />
cb3d2ecbfc97c026a0dcceb8c959b7db  TIInit_5.3.53.bts
If you run "strings" on /system/bin/bts and grep for "TII" you'll be able to tell which firmware files that version supports
<code>TIInit_3.4.27.bts<br />
TIInit_4.2.38.bts<br />
TIInit_5.2.34.bts<br />
TIInit_5.3.53.bts<br />
TIInit_6.2.31.bts</code>
*Nice picture illustrating BT architecture in Android 7:04pm CST 7/17/09*
<img src="http://farm3.static.flickr.com/2623/3997892568_5523c1b1b0_o.jpg" alt="" />
*A note for ROM devs 02:27pm CST 7/17/09*<br />
Something to note, Hero does not use any of the following legacy services and therefore they can be removed from init.rc & init.trout.rc . This is mainly something the ROM cookers should pay attention to. The btips service actually handles all of this now.</p>

<p>REMOVE THE FOLLOWING:
<code>service hcid /system/bin/hcid -s -n -f /etc/bluez/hcid.conf<br />
    socket bluetooth stream 660 bluetooth bluetooth<br />
    socket dbus_bluetooth stream 660 bluetooth bluetooth<br />
    # init.rc does not yet support applying capabilities, so run as root and<br />
    # let hcid drop uid to bluetooth with the right linux capabilities<br />
    group bluetooth net_bt_admin misc<br />
    disabled</code></p>

<p>service hciattach /system/bin/hciattach -n -s 115200 /dev/ttyHS0 texas 4000000 flow<br />
    user bluetooth<br />
    group bluetooth net_bt_admin<br />
    disabled</p>

<p>service hfag /system/bin/sdptool add --channel=10 HFAG<br />
    user bluetooth<br />
    group bluetooth net_bt_admin<br />
    disabled<br />
    oneshot</p>

<p>service hsag /system/bin/sdptool add --channel=11 HSAG<br />
    user bluetooth<br />
    group bluetooth net_bt_admin<br />
    disabled<br />
    oneshot
*Found something 01:48pm CST 7/17/09*<br />
I was looking through init.trout.rc and noticed the following lines
<code>chown bluetooth bluetooth /sys/devices/platform/msm_serial_hs.0/serial_lock_cpu<br />
chmod 0660 /sys/devices/platform/msm_serial_hs.0/serial_lock_cpu</code>
This may not seem like much but this node does not actually exist in our builds. It's possible, and probably likely, that HTC modified their kernel to support the changes that were made in the bts (btips) daemon.</p>

<p>We all are pretty much not using the HTC kernel, we're using custom compiled kernels from JAC or Cyanogen. I tried using the RUU kernel but couldn't boot at all. Is anyone able to get their phone booting off the RUU kernel and NOT one of the custom kernels that are floating around in these ROMs? If so, can you check if this device node exists?</p>

<p>I believe booting off that kernel could be the answer to the UART clock issues I'm getting and missing devices in /sys .</p>

<p>NEXT<br />
I have been toying around with the following value in init.rc that seems to affect whether or not I get an error.
<code>/proc/sys/net/unix/max_dgram_qlen</code>
The default is 10, the RUU release of Hero sets it to 999. If I change that to 10000 then it pauses the BT services and just sits there. If I revert to default I get the same error that I see when its set to 999. Wondering if there's a happy medium in queue length (qlen). Just me thinking out loud.</p>

<p>*Latest progress 11:43pm CST 7/15/09*<br />
I wanted to post some newer results I've been having with BT debugging on Hero. I found out how to circumvent the UART disable error. This is done by having the service btips statement in init.rc to look as follows
<code>service btips /system/bin/bts<br />
    socket bluetooth stream 666 bluetooth bluetooth<br />
    socket dbus_bluetooth stream 666 bluetooth bluetooth<br />
    group bluetooth net_bt_admin root misc<br />
    disabled<br />
    oneshot</code>
The most important part is "oneshot" which tells Android to NOT restart the btips service after it dies. If you leave this off then it will relaunch btips service and tie up the I2C bus.</p>

<p>The newest error I'm getting is the inability to launch HCI. This is hopefully the LAST error before I can get BT functional! Anyways, just wanted to update everyone that I have not stopped working on bluetooth.
<code>1247718990.888806 BTSTACK(778) INFO  | UATRAN: HCI Command was not acknowledged with an event<br />
[ vendor/ti/btips-linux/B_TIPS/btstack/hcitrans/uart/uarttran.c:298 ]<br />
1247718990.889935 BTSTACK(778) INFO  | HCI: HCI_Process detected transport failure<br />
[ vendor/ti/btips-linux/B_TIPS/btstack/stack/hci/hci_proc.c:1596 ]<br />
1247718990.890179 BTSTACK(778) INFO  | RADIOMGR:  RmgrHciCallback: 0x6<br />
[ vendor/ti/btips-linux/B_TIPS/btstack/stack/radiomgr.c:364 ]<br />
1247718990.890362 BTSTACK(778) INFO  | RADIOMGR:  HCI init failed (retrying)<br />
[ vendor/ti/btips-linux/B_TIPS/btstack/stack/radiomgr.c:386 ]<br />
1247718990.890484 BTSTACK(778) INFO  | RADIOMGR:  HCI init error<br />
[ vendor/ti/btips-linux/B_TIPS/btstack/stack/radiomgr.c:335 ]<br />
1247718990.890637 BTSTACK(778) INFO  | ME: HCI Init complete status: 22<br />
[ vendor/ti/btips-linux/B_TIPS/btstack/stack/me/me.c:1220 ]<br />
1247718990.890789 BTSTACK(778) INFO  | CMGR: Received event HCI_INIT_ERROR<br />
[ vendor/ti/btips-linux/B_TIPS/btstack/profiles/common/conmgr.c:591 ]<br />
1247718990.890942 BTSTACK(778) INFO  | Dbus | inside _BTBUS_COMMON_BTL_callback with event: 6 0[ vendor/ti/btips-linux/EBTIPS/apps/btbus_wrap_common.c:62 ]<br />
1247718990.893536 BTSTACK(778) INFO  | sending dbus message from  BTBUS_COMMON_BTL_callback in {vendor/ti/btips-linux/EBTIPS/apps/btbus_wrap_common.c:84}[ vendor/ti/btips-linux/EBTIPS/apps/btbus_wrap_utils.c:189 ]<br />
1247718990.898022 BTSTACK(778) INFO  | Dbus | _BTBUS_COMMON_BTL_callback signal sent: 6 0[ vendor/ti/btips-linux/EBTIPS/apps/btbus_wrap_common.c:87 ]<br />
1247718990.898358 BTSTACK(778) FATAL | HCI Init Status Received while neither FM nor BT On in progress[ vendor/ti/btips-linux/EBTIPS/btl/ti_chip_mngr/ti_chip_mngr.c:1232 ]<br />
1247718990.898541 BTSTACK(778) Assert | 0[ vendor/ti/btips-linux/EBTIPS/btl/ti_chip_mngr/ti_chip_mngr.c:1232 ]<br />
1247718990.899121 BTSTACK(778) FATAL | signal 11 sent to our program from address 0xdeadbaad and code 1[ vendor/ti/btips-linux/EBTIPS/apps/btt_task.c:102 ]</code>
I'll update this main post as I, or others, come up with progress or advancements.</p>

<p>The directories for this are already created in the latest Hero init.rc . Just need to create the ddb file
<code>touch /data/btips/TI/BtDeviceDb.ddb<br />
chmod 666 /data/btips/TI/BtDeviceDb.ddb</code>
The results of making these changes is you are able to get ALL bluetooth services and sockets created. Bluetooth is working from the commandline, just not on the frontend where we need it.</p>

<p>PS:xda那边似乎有人已经放出hero shippment rom, 蓝牙问题应该已经解决了....静待佳音吧!
<img src="http://foto.hivan.net/download.png" alt="" />:[download id="6"] | <a href="http://cid-dd052430190ebbbc.skydrive.live.com/self.aspx/Public/Android/com.alex.BluetoothFileshare.apk">skydrive 下载</a></p>
