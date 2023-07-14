# Android 简易访问Twitter,youtube,facebook方法

<p>其实就是修改hosts文件.<br />
而这个hosts文件我是已经修改完毕的...直接cat到手机内覆盖原文件就可以了!</p>

<p>执行之前请将hosts复制到sdcard的根目录,然后cmd,cd如adb目录,然后执行其下代码:</p>

<p>adb remount<br />
cat /sdcard/hosts >  /etc/hosts</p>

<p>一切搞定!
<strong>请对于hosts上的IP地址低调传播,谢谢!</strong></p>

<p>PS:本hosts修改大法已经基本完全失效,基本所有有效果的IP地址都被屏蔽,如果有条件,自己建立一个VPN吧!不过对于联通的用户我要给你一个大大的警告:联通屏蔽VPN.....</p>
