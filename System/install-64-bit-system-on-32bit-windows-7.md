# 32位windows 7 安装64位系统

<p>前言:
<ul>
	<li>我并不是为了双系统</li>
	<li>因为大部分32位程序在64位Windows 7下运行完美</li>
	<li>我是全部从新安装,并无保留任何注册表内容,是否相互兼容未测试!</li>
</ul>
32位 windows 7中安装64位windows 7会提示与此系统不兼容.</p>

<p>不过利用cmd可以正常安装.
<blockquote>x:\setup /installfrom:y:\sources\install.wim</blockquote></p>

<p>x,y为代表符,自行更改盘符."/installfrom”参数指向64位的install.wim，这里的“f:\setup”代表32位Windows 7，而“y:\sources\install.wim”则表示来自64位Windows 7的安装镜像（具体盘符根据个人电脑不同可能有差别，大家自行修改吧）.有不懂的留言!</p>

<p>PS:无法升级,两个系统相互之间不兼容!只能从新安装.<br />
PS2:AE对一个15秒的动画序列帧已经渲染了两个多小时了,极度无语.明天十点之前还要出门,打算不睡觉了!</p>
