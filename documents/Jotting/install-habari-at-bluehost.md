---
layout: post
title: Bluehost上安裝Habari
date: 2009-02-17 15:01
categories: 
- web
tags:
- bluehost
- habari
- install
published: true
comments: true
---
<p>Habari我還在試用階段...不得承認wordpress確實是一個好的博客程序,但是對於他的日進臃腫我有點微詞... 可憐我不是代碼編寫出身,所以很多問題不得不求助於人,還好身邊有一個好老師<a href="http://blog.tinyau.net/" target="_blank">"天佑"</a>給我提供了很多幫助!</p>

<p>以下就寫寫近期的一些內容.
<!--more-->
 因為bluehost原本就只願PHP5,而其他的一些標準我不太懂,但是據我安裝下來基本都已經全部滿足,唯一的就是需要自己開啟pdo for mysql. 在這之前你最好是下載一個requirements.php來測試一下你的主機是否已經為你的habari做好了準備.</p>

<p>將requirements.php上傳到自己需要安裝的文件夾下,輸入http://youblogurl/requirements.php 如果切OK,那么就會如下顯示:</p>

<p><img style="max-width: 800px" src="http://farm4.static.flickr.com/3587/3287279471_f9b8fff903.jpg?v=0" alt="" /></p>

<p>如若不然,就會有如下顯示:</p>

<p><img style="max-width: 800px" src="http://farm4.static.flickr.com/3595/3287279661_42714c7679.jpg?v=0" alt="" /></p>

<p>我這裡提示的是需要安裝或者打開pdo,如果你也是bluehost,那么基本就是這個提示了!</p>

<p>其實到這步是不需要聯繫客服幫你打開的,默認bluehost就已經安裝了pdo,主要是需要打開而已!</p>

<p>在自己的服務器上建立php.ini文檔,在CP上點擊PHP Config,然後選擇install php.ini master file 不出意外在根服務器上就已經建立php.ini了,加入如下語句:
<coolcode>extension_dir = "/usr/lib/php/extensions/no-debug-zts-20060613"<br />
extension=pdo.so extension=pdo_mysql.so</coolcode>
到這裡當然還沒有結束,需要將php.ini複製到你所要安裝的habari文件夾...</p>

<p>這步一定要做.我不知道原理,所以不要問我,我只是如此操作了,成功了.反而刪除后就會出現需要激活pdo的提示.</p>

<p>然後自然就是下載habari,上傳到安裝目錄,輸入路徑...然後就是和wordpress的安裝順序一樣了!</p>

<p><span style="color: #ff6600;">這裡要提示一點:在網上有說安裝habari要將其目錄設定為777,我的經驗是不要這樣設置,這樣會造成訪問此目錄的時候出現505錯誤頁面...<a href="http://blog.tinyau.net/" target="_blank">"天佑"</a>幫我測試的時候就是如此!...</span>後來我更改回來后就正常訪問了...</p>

<p>當然,很多人都是wordpress的用戶,所以如果你需要導入wordpress的原始數據的話需要做如下工作:
<ol>
	<li>安裝一個插件,用以取消自動保存.</li>
	<li>禁用後臺的多版本保存.</li>
	<li>刪除數據庫中多餘的文章版本.</li>
</ol>
完成一上步驟后在habari導入的時候才不會有重複文章導入! 當然,在我的habari測試中還出現了亂碼問題,不僅僅是導入的時候出現亂碼,在輸入博客標題,寫新文章都會出現中文亂碼問題,在發此文的時候這個問題還沒有得到解決,只能讓各位待續了! 再次感謝<a href="http://blog.tinyau.net/" target="_blank">"天佑"</a>的幫助!
<div class="zemanta-pixie"><img class="zemanta-pixie-img" src="http://img.zemanta.com/pixy.gif?x-id=df703edf-93a6-4276-a8b7-8dad80774d9b" alt="" /></div></p>
