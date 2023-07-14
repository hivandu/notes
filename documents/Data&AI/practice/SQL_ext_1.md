
# SQL练习1

-- UPPER 是转换大写的函数

```SQL
SELECT emp_name,salary * 12, UPPER(email) FROM employee;
```



-- 使用别名

```SQL
SELECT e.emp_name AS "姓名",
                salary * 12 AS "12月的工资",
                UPPER(email) "电子邮箱"
FROM employee AS e;
```



-- 无表查询

```SQL
SELECT 1+1;

-- Oracle 实现，dual只有一个字段且只包含一行数据
SELECT 1+1
FROM dual;

SELECT * FROM employee WHERE emp_name = '刘备';
```



-- 比较运算符

```SQL
select * from employee where sex <> '男'
select * from employee where salary BETWEEN 5000 and 7000
select * from employee where emp_name IN('刘备')
```





-- 一个时间段之后入职

```SQL
select emp_name, hire_date from employee where hire_date > DATE '2018-01-01'

SELECT emp_name,hire_date,manager from employee where manager IS NULL

SELECT emp_name, sex, salary from employee where sex = '女' AND salary > 10000

SELECT emp_name, sex, salary FROM employee WHERE emp_name = '刘备' OR emp_name = '张飞' OR emp_name = '赵云'
```



-- 短路运算 `short-circuit evaluation`

```SQL
select 'AND' FROM employee WHERE 1 = 0 AND 1/0 = 1;
SELECT 'OR' FROM employee where 1 = 1 OR 1/0 = 1;

-- NOT
select emp_id,emp_name FROM employee WHERE emp_name NOT IN('刘备','张飞','赵云')
```



-- 运算符优先级

```SQL
SELECT emp_name,dept_id,bonus
FROM employee
WHERE (dept_id = 2 OR dept_id = 3) AND bonus IS NOT NULL;
```



-- 去处重复值

```SQL
SELECT DISTINCT SEX FROM employee
SELECT DISTINCTROW sex from employee
```



/*
查找 2018 年 1 月 1 日之后入职，月薪小于 5000，并且奖金小于 1000（包括没有奖金）的员工。
*/

```SQL
SELECT emp_id,emp_name,salary,hire_date,bonus 
FROM employee 
WHERE hire_date > '2018-01-01' 
            AND salary < 5000 
            AND (bonus < 1000 OR bonus IS NULL)
```



-- LIKE运算符

```SQL
SELECT emp_id,emp_name,sex
FROM employee
WHERE emp_name LIKE '赵%'

SELECT emp_name,email
FROM employee
WHERE email NOT LIKE 'dengzh_@shuguo.com';
```



-- 转义字符

```SQL
CREATE TABLE t_like(c1 VARCHAR(20))
INSERT INTO t_like(c1) VALUES ('进度:25% 已完成')
INSERT INTO t_like(c1) VALUES ('日期:2019年5月25日')

SELECT c1
FROM t_like
WHERE c1 LIKE "%25\%%"

SELECT c1
FROM t_like
WHERE c1 LIKE "%25#%%" ESCAPE '#'
```





-- 大小写匹配

```SQL
SELECT emp_name,email
FROM employee
WHERE email LIKE 'M%'
```



-- 正则表达式  判断邮箱

```SQL
/*
以字母或者数字开头；
后面是一个或者多个字母、数组或特殊字符（ . _ - ）；
然后是一个 @ 字符；
之后包含一个或者多个字母、数组或特殊字符（ . - ）；
最后是域名，即 . 以及 2 到 4 个字母。
^[a-zA-Z0-9]+[a-zA-Z0-9._-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,4}$
*/

SELECT email FROM t_regexp
WHERE REGEXP_LIKE (email, BINARY '^[a-z0-9]+[a-z0-9._-]+@[a-z0-9.-]+\\.[a-z]{2,4}$','i');
```



-- 降序排序

```SQL
select emp_name,salary,hire_date
from employee
where dept_id = 4
ORDER BY salary DESC;
```



-- 多列排序(工资，入职先后)

```SQL
select emp_name,salary,hire_date
FROM employee
where dept_id = 4
ORDER BY salary DESC, hire_date;
```



-- 按照SELECT顺序

```SQL
SELECT emp_name, salary, hire_date
FROM employee
WHERE dept_id = 4
ORDER BY 2 desc, 3
```



**中文排序**

--  CONVERT 是一个函数，用于转换数据的字符集编码。以下转换为中文GBK字符集

```SQL
SELECT emp_name
from employee
WHERE dept_id = 4
ORDER BY CONVERT(1 USING GBK)
```



-- 空值排序

```SQL
select emp_name,bonus
from employee
where dept_id = 2
ORDER BY 2
```



-- 其他空值排序方法

```SQL
-- COALESCE函数将控制转换为一个指定的值
SELECT emp_name, COALESCE(bonus,0) AS bonus
FROM employee
where dept_id = 2
ORDER BY COALESCE(bonus,0);
```



/*
第六节练习:
查询所有的员工信息，按照员工的总收入（年薪加奖金）从高到低进行排序，总收入相同再按照姓名的拼音顺序排序。
*/

```SQL
SELECT emp_name, salary, bonus, (salary+bonus) as sum
FROM employee
WHERE dept_id >= 0
ORDER BY (salary+bonus) DESC, CONVERT(1 USING GBK)
```



-- TopN 排行榜

```SQL
-- 标准FETCH语法,此语法MySQL不支持，Oracle, PostgreSQL支持
SELECT emp_name, salary
FROM employee
ORDER BY salary DESC
OFFSET 0 ROWS
FETCH FIRST 5 ROWS ONLY;

-- LIMIT实现TOPN排行榜
SELECT emp_name, salary
FROM employee
ORDER BY salary DESC
LIMIT 5 OFFSET 0;

-- 第二种写法
SELECT emp_name,salary
FROM employee
ORDER BY salary DESC
LIMIT 0,5
```



-- SQL 实现分页查询

```SQL
SELECT emp_name, salary
FROM employee
ORDER BY salary DESC
LIMIT 10,5
```



-- 员工排名第3高

```SQL
select emp_name,salary 
FROM employee
WHERE salary = (
  select salary from employee
  ORDER BY salary DESC LIMIT 2,1)

SELECT emp_name,salary 
FROM employee
ORDER BY salary desc limit 5 offset 10;
```



/*
练习：使用LIMIT和OFFSET找出员工表中月薪排名第三高的所有员工
*/

```SQL
select emp_name, salary 
FROM employee
where salary = (
  select salary 
  FROM employee
  ORDER BY salary desc limit 1 offset 2)
```

