# A Preliminary Study of Machine Learning


## Gradient

```python
def loss(k):
    return 3 * (k ** 2) + 7 * k -10

# -b / 2a = -7 / 6

def partial(k):
    return 6 * k + 7

k = ramdom.randint(-10, 10)
alpha = 1e-3 # 0.001

for i in range(1000):
    k = k + (-1) * partial(k) * alpha
    print(k, loss(k))

```

