---
layout: post
title: "一段实现Copy的Java代码" 
date: 2012-01-12 12:16
comments: true
categories: 
- develop
tags: 
- java 
- copy 
- code
---

在DOS命中存在一个文件的复制命令(copy),例如,,将D盘中XX文件复制到D盘中并另命名为OO文件,则只要在命令行中输入copy即可完成.
<!--more-->

DOS中的命令:

     copy d:\xx.zip d:\oo.zip

而以下要实现的代码用Java完成以上功能.程序运行时也如下格式:

    java Copy 源文件 目标文件


### 程序实现:

要想输入源文件或目标文件的路径,可以通过命令行参数完成,但是此时就必须对输入的参数进行验证,如果输入的参数个数不是两个,或者输入的源文件路径不存在,则程序都应该给出错误信息并退出.\
 要完成这样的复制程序可以有以下两种方式操作:\
 将源文件中的内容全部读取到内存中,并一次性写入到目标文件中.\
 不讲源文件的内容全部读取进来,而是采用边读边写的方式.\
 而如果源文件过多过大,则会出现异常,因为内存装不下.


### 代码可能如下:

	//实现复制功能
	import java.io.*;
	public class Copy {
	    public static void main(String[] args) throws Exception{
	        if(args.length!=2){
	            System.out.println("输入的参数不正确");
	            System.out.println("例: Java Copy 源文件路径 目标文件路径");
	            System.exit(1);
	        }
	        File f1 = new File(args[0]);
	        File f2 = new File(args[1]);
	        if(!f1.exists()){
	            System.out.println("源文件不存在!");
	            System.exit(1);
	        }
	        InputStream input = null;
	        OutputStream out = null;
	        try{
	            input = new FileInputStream(f1);
	        }catch(FileNotFoundException e){
	            e.printStackTrace();
	        }
	        try{
	            out = new FileOutputStream(f2);
	            }catch(FileNotFoundException e){
	                e.printStackTrace();
	        }
	        if(input != null && out != null){
	            int temp = 0;
	            try {
	                while ((temp = input.read())!=-1){
	                    out.write(temp);
	                }
	                System.out.println("复制完成!");
	            }catch (IOException e){
	                e.printStackTrace();
	                System.out.println("复制失败!");
	            }
	            try{
	                input.close();
	                out.close();
	                
	            }catch (IOException e){
	                e.printStackTrace();
	            }
	        }
	    }
	}

### 运行情况:

    先编译:javac Copy.java
    执行 java copy d:\xx.zip d:\oo.zip

1.如果没有输入源文件或目标文件

    输入的参数不正确.
    例:java Copy 源文件路径 目标文件路径


2.输入的源文件路径不正确

    源文件不存在!

3.正确执行:java Copy d:\\xx.zip d:\\oo.zip

    复制完成!

当然,也可能是

    复制失败!

**本文原文出自: 李兴华 [《Java 开发与实战宝典》][]![image][]第十二章
JavaIO P430**

  [《Java 开发与实战宝典》]: http://www.amazon.cn/gp/product/B002IIE012/ref=as_li_tf_tl?ie=UTF8&tag=duart-23&linkCode=as2&camp=536&creative=3200&creativeASIN=B002IIE012
  [image]: http://www.assoc-amazon.cn/e/ir?t=duart-23&l=as2&o=28&a=B002IIE012