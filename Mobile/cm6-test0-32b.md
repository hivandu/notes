# CM6 test0 32b

<p>好吧,熟悉的人看到标题应该能猜到这就是Android 2.2 的CM版本.没想到这么快会出现,和之前发布的版本不同,这次编译与CM大神的版本!没想到在和儿子过生的当间就发布了...
<ul>
	<li>基于7月5日最新CM6源码编译，新增了电池百分比显示的开关，在Cyanogenmod 设置里进行更改</li>
	<li>默认关闭jit，在32a/32b等低端机上开启jit对性能没有改善，反倒更占内存，故在这个版本中关闭jit</li>
	<li>修改ADW的默认设置，使其常驻内存，改善从其它程序退回桌面的速度</li>
	<li>修改ADW的壁纸图库，用AOSP的图库替换了CM的图库</li>
	<li>修改了framework.jar，使用了geesun的代码试其支持中文运营商显示</li>
	<li>使用了最新的FRF91的GAPPS</li>
	<li>CM6自带的contact文件有不少bug，故换成了aosp的contact，虽然相对cm6的功能更少，但非常稳定</li>
	<li>重新编译了kernel，个人感觉比默认的kernel更稳定</li>
	<li>进一步汉化了framework和superuser等程序</li>
	<li>新增32a的支持，32a的用户也可以使用</li>
</ul>
Known Issues：
<ul>
	<li>第一次启动时可能会意外重启，完成设置后就不会出现这个问题</li>
	<li>相机中按0x变焦按键会使相机fc</li>
	<li>摄像无法正常使用</li>
	<li>由于Gapps都是Nexus专用的，故其素材的尺寸都很大，特别是gmail，显示出来很大，这个暂时无法解决</li>
	<li>Dream和Magic因为性能问题，不支持flash，即便刷了2.2也不可能运行flash，所以不要去市场下载flash程序了，不会起作用的，另外也不要再求使用flash的方法了，在地球上不存在解决方法，除非你换手机</li>
</ul>
App2sd：<br />
rom支持app2sd，但默认没有开启。<br />
开启方法：<br />
在超级终端中输入：<br />
su<br />
pm setInstallLocation 2<br />
即可开启app2sd，不需要有ext分区就可以使用，但官方的这个app2sd还不太稳定，不建议使用。<br />
Download：<br />
Dream/Magic 32b:http://thesoloblack.com/rom/cm6-test0-32b-0705-fixed.zip<br />
Magic 32a:http://thesoloblack.com/rom/cm6-test0-32a-0705-fixed.zip</p>

<p><strong>本更新信息和下载链接均来源于机锋网!</strong></p>
