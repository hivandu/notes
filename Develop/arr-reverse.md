---
title: 数组倒置
date: 2012-1-3 12:15:33
categories:
- develop
tags:
- java
- 数组
---

有这样一个题目,给出了一个一维整型数组,要求对其进行倒置后然后输出!有可能以下的解决办法,如果有更好的解决办法欢迎回复说明!

```
public static void reverse(int x[]){
    int foot=0;
    int head=0;
    if(x.length%2==0){
        foot=x.length/2;
            head=foot-1;
            for(int i=0; i int temp=x[head];
            x[head]=x[foot];
            x[foot]=temp;
            head--;
            foot++;
        }
    }else{
        foot=x.length/2;
        head=foot;
        for(int i=0; i int temp=x[head];
        x[head]=x[foot];
        x[foot]=temp;
        head--;
        foot++;
        }
    }
}
```