---
layout: post
title: "WordPress 插件——CoolPlayer"
date: 2006-10-16 14:21
categories: 
- web
tags:
- coolplayer
- plugin
- wordpress
published: true
comments: true
---
<p><p>该插件可以让你很方便的在日志中插入 flash，quicktime，realmedia，windows media 等各种格式的媒体文件，该插件可以自动识别媒体类型来选择播放器，而无需用特殊的标签指定，你可以设置播放器的宽、高，是否自动播放，是否循环播放，以及媒体文件名的字符集。同时可以指定多个媒体文件，并可以在多个媒体之间无刷新切换，默认就可以很好的支持中文文件名，该插件所插入的媒体不但在 Windows 下可以播放，在 Linux 下同样可以播放。</p> <!--more-->  <p><a></a></p>  <h5>客户端要求：</h5>  <p>Windows 客户端需要安装 Windows Media Player，Real Player，QuickTime，Flash Player 等播放器或者其它支持这些格式的播放器，及其相应的浏览器插件（一般安装了这些播放器以后，相应的浏览器插件就安装了）。</p>  <p>Linux 客户端需要安装 MPlayer，VLC Player，Real Player，Flash Player 播放器及其浏览器插件。</p>  <p>上面这些播放器不一定都需要安装，用户可以根据自己的习惯选择自己喜欢的播放器。更多细节请参见 <a href="http://www.coolcode.cn/?p=66">网页内嵌多媒体内容的完美实现</a> 一文。</p>  <h5>安装</h5>  <p>安装很简单，下载 <a href="http://download.coolcode.cn/coolplayer.zip">coolplayer.zip</a> 然后直接解压缩到 WordPress 的插件目录下，然后在后台激活该插件就可以了。</p>  <p>如果你想使用自己服务器上的 RPC 服务器的话，可以单独下载 <a href="http://download.coolcode.cn/coolplayer_rpc.zip">coolplayer_rpc.zip</a>，然后解压缩到 WordPress 的插件目录下，然后将 coolplayer.js 文件中的</p>  <p>coolplayer_rpc.use_service(‘http://coolcode.cn/wp-content/plugins/coolplayer/rpc.php‘);</p>  <p>这句中的路径改为你网站的中绝对路径即可。</p></p>
