# Introduction to Artificial Intelligence


> The code address of this article is: [Example 01](https://github.com/hivandu/practise/blob/master/AI-basic/example_01.ipynb)

> The source code is in ipynb format, and the output content can be viewed.

## rule based

```python
import random 
from icecream import ic


#rules = """
#复合句子 = 句子 , 连词 句子
#连词 = 而且 | 但是 | 不过
#句子 = 主语 谓语 宾语
#主语 = 你| 我 | 他 
#谓语 = 吃| 玩 
#宾语 = 桃子| 皮球
#    
#"""

rules = """
复合句子 = 句子 , 连词 复合句子 | 句子
连词 = 而且 | 但是 | 不过
句子 = 主语 谓语 宾语
主语 = 你| 我 | 他 
谓语 = 吃| 玩 
宾语 = 桃子| 皮球
    
"""

def get_grammer_by_description(description):
    rules_pattern = [r.split('=') for r in description.split('\n') if r.strip()]
    target_with_expend = [(t, ex.split('|')) for t, ex in rules_pattern]
    grammer = {t.strip(): [e.strip() for e in ex] for t, ex in target_with_expend}

    return grammer

#generated = [t for t in random.choice(grammer['句子']).split()]

#test_v = [t for t in random.choice(grammer['谓语']).split()]


def generate_by_grammer(grammer, target='句子'):
    if target not in grammer: return target

    return ''.join([generate_by_grammer(grammer, t) for t in random.choice(grammer[target]).split()])

if __name__ == '__main__':

    grammer = get_grammer_by_description(rules)

    #ic(generated)
    #ic(test_v)
    #ic(generate_by_grammer(grammer))
    ic(generate_by_grammer(grammer, target='复合句子'))
```


## water pouring

```python
def water_pouring(b1, b2, goal, start=(0, 0)):
    if goal in start: 
        return [start]

    explored = set()
    froniter = [[('init', start)]]

    while froniter:
        path = froniter.pop(0)
        (x, y) = path[-1][-1]
        
        for (state, action) in successors(x, y, b1, b2).items():
            if state not in explored:
                explored.add(state)

                path2 = path + [(action, state)]

                if goal in state:
                    return path2
                else:
                    froniter.append(path2)

    return []


def successors(x, y, X, Y):
    return {
        ((0, y+x) if x + y <= Y else (x + y - Y, Y)): 'X -> Y',
        ((x + y, 0) if x + y <= X else  (X, x + y - X)): 'X <- Y',
        (X, y): '灌满X',
        (x, Y): '灌满Y',
        (0, y): '倒空X',
        (x, 0): '倒空Y',
    }


if __name__ == '__main__':
    print(water_pouring(4, 9, 5))
    print(water_pouring(4, 9, 5, start=(4, 0)))
    print(water_pouring(4, 9, 6))
```