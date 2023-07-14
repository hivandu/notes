# Foundation of Artificial Intelligence - Lecture 1

## Algorithm --> Data Structure

> No obvious solution ==> Algorithm engineers do it
> If there is a clear implementation path ==> the person who develops the project will do it

## What's the Algorithm? 

> {Ace of hearts, 10 of spades, 3 of spades, 9 of hearts, 9 clubs, 4 of diamonds, J}

> First: Hearts> Diamonds> Spades> Clubs
> Second: Numbers are arranged from small to large

1. Some people put the colors together first
2. Some people arrange the size first, and extract the colors one by one


$$ 1024 -->  10^3 --> 1k $$
$$ 1024 * 1024 -->  10^6 --> 1M $$
$$ 1024 * 1024 * 1024 -->  10^9 --> 1G $$


```
struction-0  00011101
struction-1  00011111 
struction-2  00011100
struction-3  00011101
struction-4  00011100
struction-5  00011001
```
2.6G Hz 

```python
def fac(n): # return n!
    if n == 1: 
        return 1 # 返回操作
    else:
        return n * fac(n-1) # 乘法操作 + 返回操作 + 函数调用
```

```python
fac(1)
> 1

fac(100)
> 93326215443944152681699238856266700490715968264381621468592963895217599993229915608941463976156518286253697920827223758251185210916864000000000000000000000000

fac_100 = """93326215443944152681699238856266700490715968264381621468592963895217599993229915608941463976156518286253697920827223758251185210916864000000000000000000000000"""

len(fac_100)
> 158
```

```python
?? N --> fac(n)
# 乘法操作 + 返回操作 + 函数调用
?? (N - 1)--> fac(n-1)
?? N == 100  fac(N) 
??? 99
```

```bash
Object ` N --> fac(n)` not found.
Object ` (N - 1)--> fac(n-1)` not found.
Object ` N == 100  fac(N)` not found.
Object `? 99` not found.
```

$$ Time(N) - Time(N-1) = constant $$ 
$$ Time(N-1) - Time(N-2) = constant $$ 
$$ Time(N-2) - Time(N-3) = constant $$ 
$$ Time(2) - Time(1) = constant $$
$$ Time(N) - Time(1) == (N-1)constant $$ 
$$ Time(N) == (N-1)constant +  Time(1) $$ 
$$ Time(N) == N * constant +  (Time(1) - constant) $$ 







