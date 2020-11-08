---
layout: post
title: habari的時區和more
date: 2009-02-21 15:01
categories: 
- web
tags:
- habari
- more
- 时区
published: true
comments: true
---
<p>在這之前還一直在煩惱habari的時區問題和more代碼截斷.</p>

<p><a href="http://fireyy.com/" target="_blank">fireyy</a>給了我一些幫助.</p>

<p><!--more--></p>

<p>habari最新的版本已經加上了時區的調整.而這個版本是需要svn方式獲取才可以.基於bluehost默認沒有開通ssh方式.那么我就只有選擇等待habari下一版本的正式發布,又或者聯繫bluehost增加ssh訪問,并花一定時間來學習和熟悉svn獲取的方式!</p>

<p>另外一個就是代碼截斷.在habari總一樣可以試用more代碼來對文章進行截斷,在所用模板總加入:</p>

<p>// Only uses the <!--more--> tag, with the 'more' as the link to full post<br />
Format::apply_with_hook_params( 'more', 'post_content_out', 'more' );</p>

<p>這樣在編寫entry的時候就與wordpress的效果是相同的了.</p>

<p>而整個habari的模板代碼和wordpress是比較類似的!頭疼的是plugin和wordpress有很多不一樣的地方!所以對於不懂代碼的我來說,就無從著手了...</p>

<p>奇怪的一點是我現在說使用的habari模板激活后有一個setting的選項,點擊以後并無反應!</p>

<p>我想本身是可以修改的...</p>

<p>比較丟人的地方在於,前邊的文章總提到了歸檔頁面,其實并不是真的在頁面中直接添加代碼就可以了!而是我所用的模板中已經加入了兼容代碼...對於如何建立頁面來實現仍然不清楚!</p>

<p><span style="color: #ff6600;">
</span></p>

<p><span style="color: #ff6600;">2009-2-21 01:53 BTW:好吧,我承認我自己又傻了一回...并不是需要在服務器上安裝SVN才可以的..也可以將svn服務器上的程序下載到本地然後再上傳,效果是一樣的!而我現在說安裝的habari就更新到了0.6beta.這樣就完全解決了時區問題!並且增加了zh_tw的文件包!雖然漢化并不完全,但是已經很好了!</span></p>
