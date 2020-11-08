---
layout: post
title: 去掉wp后台编辑的文章预览
date: 2006-10-15 14:21
categories: 
- web
tags:
- wordpress
published: true
comments: true
---
<p><p>在网上搜索此类教程。我的目的比较简单，就是想加快的速度，没有出于安全问题考虑。不过这个问题要摆出来确实蛮吓人的啊。：</p> <!--more-->  <p>前面提到因为 WordPress 2.0 的编辑预览功能使用 iframe，与我使用的一个js 代码冲突，一进入编辑窗口就会跳出来。<a href="http://readnews.info/">liyuanzao</a> 告诉我可以使用 is_preview() 这个函数，我试了试发现不行。Google 了一下，发现还是个大问题。</p>  <p>因为首先搜到的是这篇文章：<a href="http://error.wordpress.com/2005/12/27/adsense-on-wordpress-20/"> AdSense on WordPress 2.0</a>，大体的中文翻译在这儿：<a href="http://blog.tinyau.net/archives/2005/12/28/adsense-on-wordpress-20/"> 在 WordPress 2.0 使用 AdSense 注意之事</a>。讲</p>  <blockquote>   <p>在 WordPress 2.0 中有一個很酷的功能名叫 Post Preview，即是在 edit / view draft 時，在文章下方會顯示一個 preview 畫面，內裡會顯示這篇文章在發表時的樣子，但如果有使用 Goolge AdSense，當 preview 時都會直接讀取 Google AdSense 廣告，但因為文章還未發表，如果跟著 Google 嘗試尋找這篇文章時，就會出現 404 Not found 的情況，有可能會被 Google 暫停你的 AdSense 戶口。</p> </blockquote>  <p>这是不是个潜在的大危险？文章中提到了使用 is_preview() 来解决，但是不少人和我一样发现这其实并不行。有人已经到 <a href="http://trac.wordpress.org/ticket/2188">WordPress 报了bug</a>，问题的表现说得很清楚。</p>  <p>如果实在着急，可以去掉预览功能，WP 支持论坛上<a href="http://wordpress.org/support/topic/53685#post-294203">给出了方法</a>：删除 /wp-admin/post.php 中 82 到 87 行。 <a href="http://yanfeng.org/blog/703/">原文地址</a></p>  <p>啊。。下边来说重点，不管是为了自己的编辑速度还是安全方面考虑，总之我们是要去掉文章预览功能是吧。其实这是一个鸡肋功能，本来编辑器就是所见即所得了。这样等于在打开后台编辑器的同时又打开了一次主页面。。。去掉的办法上边有说，第82到87行，其实我看了一下并不是这几行。不知道是不是因为版本不对，所以有些出入。我的wp是2.0.4的。。。第82到85行删除就OK了。。具体代码是：</p>  <blockquote>   <p>&lt;div id='preview' class='wrap'&gt;     <br />&lt;h2 id=&quot;preview-post&quot;&gt;&lt;?php _e('Post Preview (updated when post is saved)'); ?&gt; &lt;small class=&quot;quickjump&quot;&gt;&lt;a href=&quot;#write-post&quot;&gt;&lt;?php _e('edit &amp;uarr;'); ?&gt;&lt;/a&gt;&lt;/small&gt;&lt;/h2&gt;      <br />&lt;iframe src=&quot;&lt;?php echo add_query_arg('preview', 'true', get_permalink($post-&gt;ID)); ?&gt;&quot; width=&quot;100%&quot; height=&quot;600&quot; &gt;&lt;/iframe&gt;      <br />&lt;/div&gt;</p></blockquote></p>
