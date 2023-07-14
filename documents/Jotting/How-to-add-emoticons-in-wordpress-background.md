---
layout: post
title: 如何在WordPress后台中加入表情符号
date: 2006-10-20 14:21
categories: 
- develop
tags:
- wordpress
- 表情
published: true
comments: true
---
<p><p>本文来自<a href="http://www.zengrong.net/wp-trackback.php?p=109">zengrong</a></p>  <p>这应该是比较老的话题了，请参阅<a href="http://dark.supercn.net/index.php/81/">Smilies in WP (wp中的表情)</a></p>  <p>WordPress自带了二十多个表情符号:</p>  <p>这些符号是使用代码插入的，要记住这些代码可不太容易，因此如果将这些表情符号嵌入到后台就方便了。</p> <!--more-->  <p>   <br />方法也并不麻烦，首先下载<a href="http://www.alexking.org/blog/2004/01/24/wp-grins-clickable-smilies-hack/"> WP-Grins</a>这个插件，按正常方式安装。由于此插件并不是针对2.0开发，因此要先修改一下。找到wp-grins.php中的下面这句（大约在34行）</p>  <blockquote>   <p>$grins .= '&lt;img src=&quot;'.get_settings('siteurl').'/wp-images/smilies/'.$grin.'&quot; alt=&quot;'.$tag.'&quot; onclick=&quot;grin(\''.$tag.'\');&quot;/&gt; ';</p> </blockquote>  <p>将其中的“/wp-images/smilies/”    <br />修改为“/wp-includes/images/smilies/”。</p>  <p>安装后在插件管理器中激活它，然后编辑“/wp-admin/admin-functions.php”文件，搜索“ edToolbar();”，大约在1079行，找到下面这句</p>  <blockquote>   <p>if (!strstr($_SERVER['HTTP_USER_AGENT'], 'Safari'))      <br />echo '       <br />&lt;div id=&quot;quicktags&quot;&gt;       <br />&lt;script src=&quot;../wp-includes/js/quicktags.js&quot;       <br />type=&quot;text/javascript&quot;&gt;&lt;/script&gt;       <br />&lt;script type=&quot;text/javascript&quot;&gt;if ( typeof tinyMCE ==       <br />&quot;undefined&quot; || tinyMCE.configs.length &lt; 1 ) edToolbar();&lt;/script&gt;       <br />&lt;/div&gt;       <br />';</p> </blockquote>  <p>再上面这句之下加入：</p>  <blockquote>   <p>if (function_exists(&quot;wp_grins&quot;)) { echo &quot;&quot;; wp_grins(); }&#160;&#160;&#160; //zrong added</p> </blockquote>  <p>修改完毕上传，打开后台撰写文章即可看到，可爱的表情已经加入到编辑器中了。</p>  <p>如果要将这些表情也加入到评论页面中，可以在主模版的comments.php中搜索如下语句（可能不完全相同）：</p>  <blockquote>   <p>&lt;p&gt;      <br />&lt;textarea class=&quot;textform&quot; name=&quot;comment&quot; id=&quot;comment&quot; cols=&quot;100%&quot;rows=&quot;10&quot; tabindex=&quot;4&quot;&gt;&lt;/textarea&gt;       <br />&lt;/p&gt;</p> </blockquote>  <p>在其上加入如下PHP语句即可：</p>  <blockquote>   <p>&lt;?php wp_grins(); ?&gt;</p> </blockquote>  <p>完毕。记下来也方便自己查找。    <br />顺便：</p>  <p>如果想在页面中加入表情，还可以借助<a href="http://www.coolcode.cn/?p=74">Emotions</a>这个插件。</p></p>
