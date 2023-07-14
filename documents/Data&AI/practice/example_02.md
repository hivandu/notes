# Initial exploration of machine learning



> The code address of this article is: [Example 02](https://github.com/hivandu/practise/blob/master/AI-basic/example_02.ipynb)

> The source code is in ipynb format, and the output content can be viewed.

##  Gradient

```python
import random

def loss(k):
    return 3 * (k ** 2) + 7 * k - 10
  
# -b / 2a = -7 / 6

def partial(k):
    return 6 * k + 7

k = random.randint(-10, 10)
alpha = 1e-3 # 0.001

for i in range(1000):
    k = k + (-1) * partial(k) *alpha
    print(k, loss(k))
    
# out
"""
7.959 124.32404299999999
-7.918246 122.66813714954799
show more (open the raw output data in a text editor) ...
-1.1833014444482555 -14.082503185837805
"""
```



## Cutting Problem

All the dynamic programming:

1. sub-problems
2. Overlapping sub-problems
3. parse solution

```python
from collections import defaultdict
from functools import lru_cache
# least recent used

prices = [1, 5, 8, 9, 10, 17, 17, 20, 24, 30, 33]
complete_price = defaultdict(int)
for i, p in enumerate(prices): complete_price[i+1] = p
  
solution = {}

cache = {}
#<- if when n .... is huge. size(cache)
# keep most important information.

@lru_cache(maxsize=2**10)
def r(n):
    # a very classical dynamic programming problem
    # if n in cache: return cache[n]

    candidates = [(complete_price[n], (n, 0))] + \
                 [(r(i) + r(n-i), (i, n - i)) for i in range(1, n)]

    optimal_price, split = max(candidates)

    solution[n] = split

    # cache[n] = optimal_price

    return optimal_price


def parse_solution(n, cut_solution):
    left, right = cut_solution[n]

    if left == 0 or right == 0: return [left+right, ]
    else:
        return parse_solution(left, cut_solution) + parse_solution(right, cut_solution)

if __name__ == '__main__':
    print(r(19))
    print(parse_solution(19, solution))
    
# out
"""
55
[11, 6, 2]
"""
```



## Dynamic

```python
from collections import defaultdict
from functools import wraps
from icecream import ic

original_price = [1,5,8,9,10,17,17,20,24,30,33]
price = defaultdict(int)

for i, p in enumerate(original_price):
    price[i+1] = p
    
def memo(func):
    cache = {}
    @wraps(func)
    def _wrap(n):
        if n in cache: result = cache[n]
        else:
            result = func(n)
            cache[n] = result
        return result
    return _wrap
  
@memo
def r(n):
    max_price, split_point = max(
        [(price[n],0)] + [(r(i) + r(n-i), i) for i in range(1, n)], key=lambda x: x[0]
    )
    solution[n]  = (split_point, n-split_point)
    return max_price
  
def not_cut(split): return split == 0
def parse_solution(target_length, revenue_solution):
    left, right = revenue_solution[target_length]
    if not_cut(left): return [right]
    return parse_solution(left, revenue_solution) + parse_solution(right, revenue_solution)
  
solution = {}
r(50)
ic(parse_solution(20,solution))
ic(parse_solution(19,solution))
ic(parse_solution(27,solution))

# out
"""
ic| parse_solution(20,solution): [10, 10]
ic| parse_solution(19,solution): [2, 6, 11]
ic| parse_solution(27,solution): [6, 10, 11]
[6, 10, 11]
"""
```



## Gradient descent

```python
import numpy as np
import matplotlib.pyplot as plt
import random

from icecream import ic

def func(x):
    return 10 * x**2 + 32*x + 9

def gradient(x):
    return 20 *x + 32
  
x = np.linspace(-10, 10)
steps = []
x_star = random.choice(x)
alpha = 1e-3

for i in range(100):
    x_star = x_star + -1*gradient(x_star)*alpha
    steps.append(x_star)

    ic(x_star, func(x_star))

fig, ax = plt.subplots()
ax.plot(x, func(x))

"""
ic| x_star: 9.368, func(x_star): 1186.3702400000002
ic| x_star: 9.14864, func(x_star): 1138.732618496
show more (open the raw output data in a text editor) ...
ic| x_star: -0.1157435825983131, func(x_star): 5.430171125980905
[<matplotlib.lines.Line2D at 0x7fd6d19545d0>]
"""

for i, s in enumerate(steps):
    ax.annotate(str(i+1), (s, func(s)))
    
plt.show()
```

![image-20210830234709856](http://qiniu.hivan.me/picGo/20210830234710.png?imgNote)







## k-means-finding-centers


### K-means

```python
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['FangSong'] # Specify the default font
mpl.rcParams['axes.unicode_minus'] = False # Solve the problem that the minus sign'-' is displayed as a square in the saved image

coordination_source = """
{name:'兰州', geoCoord:[103.73, 36.03]},
{name:'嘉峪关', geoCoord:[98.17, 39.47]},
{name:'西宁', geoCoord:[101.74, 36.56]},
{name:'成都', geoCoord:[104.06, 30.67]},
{name:'石家庄', geoCoord:[114.48, 38.03]},
{name:'拉萨', geoCoord:[102.73, 25.04]},
{name:'贵阳', geoCoord:[106.71, 26.57]},
{name:'武汉', geoCoord:[114.31, 30.52]},
{name:'郑州', geoCoord:[113.65, 34.76]},
{name:'济南', geoCoord:[117, 36.65]},
{name:'南京', geoCoord:[118.78, 32.04]},
{name:'合肥', geoCoord:[117.27, 31.86]},
{name:'杭州', geoCoord:[120.19, 30.26]},
{name:'南昌', geoCoord:[115.89, 28.68]},
{name:'福州', geoCoord:[119.3, 26.08]},
{name:'广州', geoCoord:[113.23, 23.16]},
{name:'长沙', geoCoord:[113, 28.21]},
//{name:'海口', geoCoord:[110.35, 20.02]},
{name:'沈阳', geoCoord:[123.38, 41.8]},
{name:'长春', geoCoord:[125.35, 43.88]},
{name:'哈尔滨', geoCoord:[126.63, 45.75]},
{name:'太原', geoCoord:[112.53, 37.87]},
{name:'西安', geoCoord:[108.95, 34.27]},
//{name:'台湾', geoCoord:[121.30, 25.03]},
{name:'北京', geoCoord:[116.46, 39.92]},
{name:'上海', geoCoord:[121.48, 31.22]},
{name:'重庆', geoCoord:[106.54, 29.59]},
{name:'天津', geoCoord:[117.2, 39.13]},
{name:'呼和浩特', geoCoord:[111.65, 40.82]},
{name:'南宁', geoCoord:[108.33, 22.84]},
//{name:'西藏', geoCoord:[91.11, 29.97]},
{name:'银川', geoCoord:[106.27, 38.47]},
{name:'乌鲁木齐', geoCoord:[87.68, 43.77]},
{name:'香港', geoCoord:[114.17, 22.28]},
{name:'澳门', geoCoord:[113.54, 22.19]}
"""
```

### Feacutre Extractor 

```python
city_location = {
    '香港': (114.17, 22.28)
}
test_string = "{name:'兰州', geoCoord:[103.73, 36.03]},"

import re

pattern = re.compile(r"name:'(\w+)',\s+geoCoord:\[(\d+.\d+),\s(\d+.\d+)\]")

for line in coordination_source.split('\n'):
    city_info = pattern.findall(line)
    if not city_info: continue
    
    # following: we find the city info
    
    city, long, lat = city_info[0]
    
    long, lat = float(long), float(lat)
    
    city_location[city] = (long, lat)

city_location

# output
"""
{'香港': (114.17, 22.28),
 '兰州': (103.73, 36.03),
show more (open the raw output data in a text editor) ...
 '澳门': (113.54, 22.19)}
"""

import math

def geo_distance(origin, destination):
    """
    Calculate the Haversine distance.

    Parameters
    ----------
    origin : tuple of float
        (lat, long)
    destination : tuple of float
        (lat, long)

    Returns
    -------
    distance_in_km : float

    Examples
    --------
    >>> origin = (48.1372, 11.5756)  # Munich
    >>> destination = (52.5186, 13.4083)  # Berlin
    >>> round(distance(origin, destination), 1)
    504.2
    """
    lon1, lat1 = origin
    lon2, lat2 = destination
    radius = 6371  # km

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) * math.sin(dlat / 2) +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) * math.sin(dlon / 2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = radius * c

    return d
```


### Vector Distances

+ 余弦距离 Cosine Distance
+ 欧几里得距离 Euclidean Distance
+ 曼哈顿距离 Manhattan distance or Manhattan length


```python
import matplotlib.pyplot as plt
import networkx as nx
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline

# set plt, show chinese
plt.rcParams['font.sans-serif']  = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus']  = False

city_graph = nx.Graph()
city_graph.add_nodes_from(list(city_location.keys()))
nx.draw(city_graph, city_location, with_labels=True, node_size=30)
```

![image-20210830234858640](http://qiniu.hivan.me/picGo/20210830234858.png?imgNote)



### K-means: Initial k random centers

```python
k = 10
import random
all_x = []
all_y = []

for _, location in city_location.items():
    x, y = location
    
    all_x.append(x)
    all_y.append(y)

def get_random_center(all_x, all_y):
    r_x = random.uniform(min(all_x), max(all_x))
    r_y = random.uniform(min(all_y), max(all_y))
    
    return r_x, r_y

get_random_center(all_x, all_y)

# output
"""
(93.61182991130997, 37.01816228131414)
"""

K = 5
centers = {'{}'.format(i+1): get_random_center(all_x, all_y) for i in range(K)}

from collections import defaultdict

closet_points = defaultdict(list)
for x, y, in zip(all_x, all_y):
    closet_c, closet_dis = min([(k, geo_distance((x, y), centers[k])) for k in centers], key=lambda t: t[1])    
    
    closet_points[closet_c].append([x, y])

import numpy as np

def iterate_once(centers, closet_points, threshold=5):
    have_changed = False
    
    for c in closet_points:
        former_center = centers[c]

        neighbors = closet_points[c]

        neighbors_center = np.mean(neighbors, axis=0)

        if geo_distance(neighbors_center, former_center) > threshold:
            centers[c] = neighbors_center
            have_changed = True
        else:
            pass ## keep former center
        
    return centers, have_changed

def kmeans(Xs, k, threshold=5):
    all_x = Xs[:, 0]
    all_y = Xs[:, 1]
    
    K = k
    centers = {'{}'.format(i+1): get_random_center(all_x, all_y) for i in range(K)}
    changed = True
    
    while changed:
        closet_points = defaultdict(list)

        for x, y, in zip(all_x, all_y):
            closet_c, closet_dis = min([(k, geo_distance((x, y), centers[k])) for k in centers], key=lambda t: t[1])    
            closet_points[closet_c].append([x, y])   
            
        centers, changed = iterate_once(centers, closet_points, threshold)
        print('iteration')

    return centers

kmeans(np.array(list(city_location.values())), k=5, threshold=5)

# output
"""
iteration
iteration
iteration
iteration
iteration
{'1': array([99.518, 38.86 ]),
 '2': array([117.833,  39.861]),
 '3': array([91.11, 29.97]),
 '4': array([106.81,  27.  ]),
 '5': array([116.87166667,  27.6275    ])}

"""

plt.scatter(all_x, all_y)
plt.scatter(*zip(*centers.values()))
```

![image-20210830235114060](http://qiniu.hivan.me/picGo/20210830235114.png?imgNote)



```python
for c, points in closet_points.items():
    plt.scatter(*zip(*points))
```



![image-20210830235135375](http://qiniu.hivan.me/picGo/20210830235135.png?imgNote)

```python
city_location_with_station = {
    '能源站-{}'.format(i): position for i, position in centers.items()
}
city_location_with_station

# output
"""
{'能源站-1': (108.82946246581274, 26.05763939719317),
 '能源站-2': (97.96769355736322, 22.166113183141032),
 '能源站-3': (114.05390380408154, 38.7698708467224),
 '能源站-4': (118.49242085311417, 28.665716162786204),
 '能源站-5': (125.08287617496866, 25.55784683330647)}
"""
def draw_cities(citise, color=None):
    city_graph = nx.Graph()
    city_graph.add_nodes_from(list(citise.keys()))
    nx.draw(city_graph, citise, node_color=color, with_labels=True, node_size=30)

%matplotlib inline

plt.figure(1,figsize=(12,12)) 
draw_cities(city_location_with_station, color='green')
draw_cities(city_location, color='red')
```

![image-20210830235243564](http://qiniu.hivan.me/picGo/20210830235243.png?imgNote)

## About the dataset
> This contains data of news headlines published over a period of 15 years. From the reputable Australian news source ABC (Australian Broadcasting Corp.)
Site: http://www.abc.net.au/
Prepared by Rohit Kulkarni

```python
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline


data = pd.read_csv("./data/abcnews-date-text.csv",error_bad_lines=False,usecols =["headline_text"])
data.head()

# output
"""
headline_text
0	aba decides against community broadcasting lic...
1	act fire witnesses must be aware of defamation
2	a g calls for infrastructure protection summit
3	air nz staff in aust strike for pay rise
4	air nz strike to affect australian travellers
"""

data.to_csv('abcnews.csv', index=False, encoding='utf8')
data.info()

# output
"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1103665 entries, 0 to 1103664
Data columns (total 1 columns):
 #   Column         Non-Null Count    Dtype 
---  ------         --------------    ----- 
 0   headline_text  1103665 non-null  object
dtypes: object(1)
memory usage: 8.4+ MB
"""
```


## Deleting dupliate headlines(if any)

```python
data[data['headline_text'].duplicated(keep=False)].sort_values('headline_text').head(8)
data = data.drop_duplicates('headline_text')
```

## NLP 

### Preparing data for vectorizaion
However, when doing natural language processing, words must be converted into vectors that machine learning algorithms can make use of. If your goal is to do machine learning on text data, like movie reviews or tweets or anything else, you need to convert the text data into numbers. This process is sometimes referred to as “embedding” or “vectorization”.

In terms of vectorization, it is important to remember that it isn’t merely turning a single word into a single number. While words can be transformed into numbers, an entire document can be translated into a vector. Not only can a vector have more than one dimension, but with text data vectors are usually high-dimensional. This is because each dimension of your feature data will correspond to a word, and the language in the documents you are examining will have thousands of words.

### TF-IDF
In information retrieval, tf–idf or TFIDF, short for term frequency–inverse document frequency, is a numerical statistic that is intended to reflect how important a word is to a document in a collection or corpus. It is often used as a weighting factor in searches of information retrieval, text mining, and user modeling. The tf-idf value increases proportionally to the number of times a word appears in the document and is offset by the frequency of the word in the corpus, which helps to adjust for the fact that some words appear more frequently in general. Nowadays, tf-idf is one of the most popular term-weighting schemes; 83% of text-based recommender systems in the domain of digital libraries use tf-idf.

Variations of the tf–idf weighting scheme are often used by search engines as a central tool in scoring and ranking a document's relevance given a user query. tf–idf can be successfully used for stop-words filtering in various subject fields, including text summarization and classification.

One of the simplest ranking functions is computed by summing the tf–idf for each query term; many more sophisticated ranking functions are variants of this simple model.

```python
punc = ['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}',"%"]
stop_words = text.ENGLISH_STOP_WORDS.union(punc)
desc = data['headline_text'].values
vectorizer = TfidfVectorizer(stop_words = stop_words)
X = vectorizer.fit_transform(desc)

word_features = vectorizer.get_feature_names()
print(len(word_features))
print(word_features[5000:5100])

# output
"""
96397
['abyss', 'ac', 'aca', 'acacia', 'acacias', 'acadamy', 'academia', 'academic', 'academics', 'academies', 'academy', 'academys', 'acai', 'acapulco', 'acars', 'acason', 'acasuso', 'acb', 'acbf', 'acc', 'acca', 'accan', 'accc', 'acccc', 'acccs', 'acccused', 'acce', 'accedes', 'accelerant', 'accelerants', 'accelerate', 'accelerated', 'accelerates', 'accelerating', 'acceleration', 'accelerator', 'accen', 'accent', 'accents', 'accentuate', 'accentuates', 'accentuating', 'accenture', 'accept', 'acceptability', 'acceptable', 'acceptably', 'acceptance', 'acceptances', 'accepted', 'accepting', 'acceptor', 'acceptors', 'accepts', 'accerate', 'acces', 'access', 'accessary', 'accessed', 'accesses', 'accessibility', 'accessible', 'accessing', 'accessories', 'accessory', 'accesss', 'acci', 'accid', 'accide', 'acciden', 'accidenatlly', 'accidenbt', 'accident', 'accidental', 'accidentally', 'accidently', 'accidents', 'acciona', 'accis', 'acclaim', 'acclaimed', 'acclamation', 'acclimatise', 'acco', 'accolade', 'accolades', 'accom', 'accomm', 'accommoda', 'accommodate', 'accommodated', 'accommodates', 'accommodating', 'accommodation', 'accomo', 'accomodation', 'accomommodation', 'accompanied', 'accompanies', 'accompaniment']
"""
```

### Stemming
Stemming is the process of reducing a word into its stem, i.e. its root form. The root form is not necessarily a word by itself, but it can be used to generate words by concatenating the right suffix. For example, the words fish, fishes and fishing all stem into fish, which is a correct word. On the other side, the words study, studies and studying stems into studi, which is not an English word.

### Tokenizing
Tokenization is breaking the sentence into words and punctuation,

```python
stemmer = SnowballStemmer('english')
tokenizer = RegexpTokenizer(r'[a-zA-Z\']+')

def tokenize(text):
    return [stemmer.stem(word) for word in tokenizer.tokenize(text.lower())]
```

**Vectorization with stop words(words irrelevant to the model), stemming and tokenizing**

```python
vectorizer2 = TfidfVectorizer(stop_words = stop_words, tokenizer = tokenize)
X2 = vectorizer2.fit_transform(desc)
word_features2 = vectorizer2.get_feature_names()
print(len(word_features2))
print(word_features2[:50]) 

# output
"""
65232
["'a", "'i", "'s", "'t", 'aa', 'aaa', 'aaahhh', 'aac', 'aacc', 'aaco', 'aacta', 'aad', 'aadmi', 'aag', 'aagaard', 'aagard', 'aah', 'aalto', 'aam', 'aamer', 'aami', 'aamodt', 'aandahl', 'aant', 'aap', 'aapa', 'aapt', 'aar', 'aaradhna', 'aardman', 'aardvark', 'aargau', 'aaron', 'aaronpaul', 'aarwun', 'aat', 'ab', 'aba', 'abaaoud', 'ababa', 'aback', 'abadi', 'abadon', 'abal', 'abalon', 'abalonv', 'abama', 'abandon', 'abandond', 'abandong']
"""

vectorizer3 = TfidfVectorizer(stop_words = stop_words, tokenizer = tokenize, max_features = 1000)
X3 = vectorizer3.fit_transform(desc)
words = vectorizer3.get_feature_names()
```

For this, we will use k-means clustering algorithm.
### K-means clustering
(Source [Wikipedia](https://en.wikipedia.org/wiki/K-means_clustering#Standard_algorithm))

### Elbow method to select number of clusters

This method looks at the percentage of variance explained as a function of the number of clusters: One should choose a number of clusters so that adding another cluster doesn't give much better modeling of the data. More precisely, if one plots the percentage of variance explained by the clusters against the number of clusters, the first clusters will add much information (explain a lot of variance), but at some point the marginal gain will drop, giving an angle in the graph. The number of clusters is chosen at this point, hence the "elbow criterion". This "elbow" cannot always be unambiguously identified. Percentage of variance explained is the ratio of the between-group variance to the total variance, also known as an F-test. A slight variation of this method plots the curvature of the within group variance.

#### **Basically, number of clusters = the x-axis value of the point that is the corner of the "elbow"(the plot looks often looks like an elbow)**

```python
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(X3)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.savefig('elbow.png')
plt.show()
```

![image-20210830235601231](http://qiniu.hivan.me/picGo/20210830235601.png?imgNote)



As more than one elbows have been generated, I will have to select right amount of clusters by trial and error. So, I will showcase the results of different amount of clusters to find out the right amount of clusters.

```python
print(words[250:300])

# output
"""
['decis', 'declar', 'defenc', 'defend', 'delay', 'deliv', 'demand', 'deni', 'despit', 'destroy', 'detent', 'develop', 'die', 'director', 'disabl', 'disast', 'discuss', 'diseas', 'dismiss', 'disput', 'doctor', 'dog', 'dollar', 'domest', 'donald', 'donat', 'doubl', 'doubt', 'draw', 'dri', 'drink', 'drive', 'driver', 'drop', 'drought', 'drown', 'drug', 'drum', 'dump', 'dure', 'e', 'eagl', 'earli', 'eas', 'east', 'econom', 'economi', 'edg', 'educ', 'effort']
"""
```



### 3 Clusters

```python
kmeans = KMeans(n_clusters = 3, n_init = 20, n_jobs = 1) # n_init(number of iterations for clsutering) n_jobs(number of cpu cores to use)
kmeans.fit(X3)
# We look at 3 the clusters generated by k-means.
common_words = kmeans.cluster_centers_.argsort()[:,-1:-26:-1]
for num, centroid in enumerate(common_words):
    print(str(num) + ' : ' + ', '.join(words[word] for word in centroid))
    
# output
"""
0 : new, say, plan, win, council, govt, australia, report, kill, fund, urg, court, warn, water, australian, nsw, open, chang, year, qld, interview, wa, death, face, crash
1 : polic, investig, probe, man, search, offic, hunt, miss, arrest, death, car, shoot, drug, seek, attack, assault, say, murder, crash, charg, driver, suspect, fatal, raid, station
2 : man, charg, murder, court, face, jail, assault, stab, die, death, drug, guilti, child, sex, accus, attack, woman, crash, arrest, car, kill, miss, sydney, alleg, plead
"""
```



###  5 Clusters

```python
kmeans = KMeans(n_clusters = 5, n_init = 20, n_jobs = 1)
kmeans.fit(X3)
# We look at 5 the clusters generated by k-means.
common_words = kmeans.cluster_centers_.argsort()[:,-1:-26:-1]
for num, centroid in enumerate(common_words):
    print(str(num) + ' : ' + ', '.join(words[word] for word in centroid))

# output
"""
0 : man, plan, charg, court, govt, australia, face, murder, accus, jail, assault, stab, urg, drug, death, attack, child, sex, die, woman, guilti, say, alleg, told, car
1 : new, zealand, law, year, plan, open, polic, home, hospit, centr, deal, set, hope, australia, look, appoint, announc, chief, say, south, minist, govt, rule, servic, welcom
2 : say, win, kill, report, australian, warn, interview, open, water, fund, nsw, crash, death, urg, year, chang, wa, sydney, claim, qld, hit, attack, world, set, health
3 : council, plan, consid, fund, rate, urg, seek, new, merger, water, land, develop, reject, say, mayor, vote, chang, elect, rise, meet, park, push, want, govt, approv
4 : polic, investig, man, probe, search, offic, hunt, miss, arrest, death, car, charg, shoot, drug, seek, attack, assault, murder, crash, say, driver, fatal, suspect, raid, woman
"""
```



###  6 Clusters

```python
kmeans = KMeans(n_clusters = 6, n_init = 20, n_jobs = 1)
kmeans.fit(X3)
# We look at 6 the clusters generated by k-means.
common_words = kmeans.cluster_centers_.argsort()[:,-1:-26:-1]
for num, centroid in enumerate(common_words):
    print(str(num) + ' : ' + ', '.join(words[word] for word in centroid))
    
# output
"""
0 : council, govt, australia, report, warn, urg, fund, australian, water, nsw, chang, qld, wa, health, elect, rural, countri, hour, sa, boost, climat, govern, servic, south, consid
1 : man, charg, murder, court, face, jail, assault, stab, die, death, drug, guilti, child, sex, accus, attack, woman, crash, arrest, car, kill, miss, sydney, plead, alleg
2 : polic, investig, probe, man, search, offic, hunt, miss, arrest, death, car, shoot, drug, seek, attack, crash, assault, murder, charg, driver, say, fatal, suspect, raid, warn
3 : win, kill, court, interview, crash, open, death, sydney, face, year, claim, hit, attack, world, set, final, day, hous, die, home, jail, talk, return, cup, hospit
4 : new, zealand, law, year, plan, open, council, polic, home, hospit, centr, deal, set, hope, australia, appoint, look, announc, chief, say, govt, south, minist, mayor, welcom
5 : say, plan, council, govt, water, need, group, chang, labor, minist, govern, opposit, public, mp, health, union, green, hous, develop, resid, report, expert, cut, australia, mayor
"""
```



###  8 Clusters

```python
kmeans = KMeans(n_clusters = 8, n_init = 20, n_jobs = 1)
kmeans.fit(X3)
# Finally, we look at 8 the clusters generated by k-means.
common_words = kmeans.cluster_centers_.argsort()[:,-1:-26:-1]
for num, centroid in enumerate(common_words):
    print(str(num) + ' : ' + ', '.join(words[word] for word in centroid))
    
# output
"""
0 : polic, say, man, miss, arrest, jail, investig, car, search, murder, attack, crash, kill, probe, die, hunt, shoot, assault, offic, drug, stab, accus, fatal, guilti, bodi
1 : death, hous, polic, toll, investig, man, probe, inquest, rise, woman, coron, blaze, price, public, white, babi, sentenc, famili, road, spark, jail, prompt, blame, custodi, report
2 : plan, council, govt, water, new, say, develop, hous, group, chang, unveil, reject, park, urg, centr, public, expans, green, resid, health, reveal, labor, govern, opposit, power
3 : court, face, man, accus, told, hear, murder, high, case, appear, rule, charg, alleg, appeal, drug, jail, woman, death, assault, order, sex, stab, challeng, teen, polic
4 : australia, govt, kill, report, warn, australian, urg, fund, nsw, interview, water, open, crash, qld, chang, wa, year, day, claim, hit, attack, sydney, set, health, world
5 : new, council, zealand, law, fund, year, consid, water, urg, open, say, seek, rate, centr, mayor, govt, elect, look, develop, land, deal, hope, set, push, home
6 : win, award, cup, titl, open, gold, stage, world, final, tour, elect, australia, lead, seri, aussi, claim, second, australian, big, england, grand, m, battl, race, record
7 : charg, man, murder, face, assault, drug, polic, child, sex, woman, teen, death, stab, drop, alleg, attack, rape, men, guilti, shoot, bail, sydney, fatal, driver, yo
"""
```

Because even I didn't know what kind of clusters would be generated, I will describe them in comments.

## Other discussions

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
import gensim
from collections import Counter
import string
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
import pyLDAvis.gensim_models
from wordcloud import WordCloud, STOPWORDS
from textblob import TextBlob
from spacy import displacy
import nltk

import warnings

warnings.filterwarnings('ignore')

# set plt
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'figure.figsize': [16, 12]})
# plt.figure(figsize = [20, 20])
plt.style.use('seaborn-whitegrid')

df = pd.read_csv('../data/abcnews-date-text.csv', nrows = 10000)
df.head()

# output
"""
publish_date	headline_text
0	20030219	aba decides against community broadcasting lic...
1	20030219	act fire witnesses must be aware of defamation
2	20030219	a g calls for infrastructure protection summit
3	20030219	air nz staff in aust strike for pay rise
4	20030219	air nz strike to affect australian travellers
"""
```

The data set contains only two columns, the release date and the news title.

For simplicity, I will explore the first 10,000 rows in this dataset. Since the titles are sorted by publish_date, they are actually two months from February 19, 2003 to April 7, 2003.



###  Number of characters present in each sentence

Visualization of text statistics is a simple but insightful technique.

They include:

Word frequency analysis, sentence length analysis, average word length analysis, etc.

These really help to explore the basic characteristics of text data.

For this, we will mainly use histograms (continuous data) and bar graphs (categorical data).

First, let me look at the number of characters in each sentence. This can give us a rough idea of the length of news headlines.

```python
df['headline_text'].str.len().hist()
```

![image-20210831000016641](http://qiniu.hivan.me/picGo/20210831000016.png?imgNote)



### number of words appearing in each news headline

The histogram shows that the range of news headlines is 10 to 70 characters, usually between 25 and 55 characters.

Now, we will continue to explore the data verbatim. Let's plot the number of words that appear in each news headline.

```python
df['headline_text'].str.split().map(lambda x: len(x)).hist()
```

![image-20210831000044656](http://qiniu.hivan.me/picGo/20210831000044.png?imgNote)



###  Analysing word length

Obviously, the number of words in news headlines is in the range of 2 to 12, and most of them are between 5 and 7.

Next, let's check the average word length in each sentence.

```python
df['headline_text'].str.split().apply(lambda x : [len(i) for i in x]).map(lambda x: np.mean(x)).hist()
```

![image-20210831000111622](http://qiniu.hivan.me/picGo/20210831000111.png?imgNote)

The average word length is between 3 and 9, and the most common length is 5. Does this mean that people use very short words in news headlines?

Let us find out.

One reason that may not be the case is stop words. Stop words are the most commonly used words in any language (such as "the", "a", "an", etc.). Since the length of these words may be small, these words may cause the above graphics to be skewed to the left.

Analyzing the number and types of stop words can give us some in-depth understanding of the data.

To get a corpus containing stop words, you can use the [nltk library](https://www.nltk.org/?ref=hackernoon.com). Nltk contains stop words from multiple languages. Since we only deal with English news, I will filter English stop words from the corpus.

### Analysing stopwords

```python
# Fetch stopwords
import nltk
nltk.download('stopwords')
stop=set(stopwords.words('english'))

# output
"""
[nltk_data] Downloading package stopwords to /Users/xx/nltk_data...
[nltk_data]   Package stopwords is already up-to-date!
"""


# Create corpus
corpus=[]
new= df['headline_text'].str.split()
new=new.values.tolist()
corpus=[word for i in new for word in i]

from collections import defaultdict
dic=defaultdict(int)
for word in corpus:
    if word in stop:
        dic[word]+=1
        
# Plot top stopwords

top=sorted(dic.items(), key=lambda x:x[1],reverse=True)[:10] 
x,y=zip(*top)
plt.bar(x,y)
```

Draw popular stop words



![image-20210831000213620](http://qiniu.hivan.me/picGo/20210831000213.png?imgNote)



### Most common words

We can clearly see that in the news headlines, stop words such as "to", "in" and "for" dominate.

So now that we know which stop words appear frequently in our text, let's check which words other than these stop words appear frequently.

We will use the counter function in the collection library to count the occurrence of each word and store it in a list of tuples. This is a very useful feature when we are dealing with word-level analysis in natural language processing.

```python
counter=Counter(corpus)
most=counter.most_common()

x, y=[], []
for word,count in most[:40]:
    if (word not in stop):
        x.append(word)
        y.append(count)
        
sns.barplot(x=y,y=x)
```

![image-20210831000244040](http://qiniu.hivan.me/picGo/20210831000244.png?imgNote)

Wow! In the past 15 years, "America", "Iraq" and "War" have dominated the headlines.

"We" here may mean the United States or us (you and me). We are not a stop word, but when we look at the other words in the picture, they are all related to the United States-the Iraq War and "we" here may mean the United States.



##  Ngram analysis

Ngram is a continuous sequence of n words. For example, "Riverbank", "Three Musketeers" and so on.
If the number of words is two, it is called a double word. For 3 characters, it is called a trigram, and so on.

Viewing the most common n-grams can give you a better understanding of the context in which the word is used.

###  Bigram analysis

To build our vocabulary, we will use Countvectorizer. Countvectorizer is a simple method for labeling, vectorizing and representing corpora in an appropriate form. Can [be found](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html?ref=hackernoon.com) in [sklearn.feature_engineering.text](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html?ref=hackernoon.com)

Therefore, we will analyze the top news in all news headlines.

```python
def get_top_ngram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(n, n)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) 
                  for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:10]

top_n_bigrams=get_top_ngram(df['headline_text'],2)[:10]
x,y=map(list,zip(*top_n_bigrams))
sns.barplot(x=y,y=x)
```

![image-20210831000322862](http://qiniu.hivan.me/picGo/20210831000322.png?imgNote)

### Trigram analysis

We can observe that dualisms such as "anti-war" and "killed" related to war dominate the headlines.

How about triples?

```python
top_tri_grams=get_top_ngram(df['headline_text'],n=3)
x,y=map(list,zip(*top_tri_grams))
sns.barplot(x=y,y=x)
```

![image-20210831000347490](http://qiniu.hivan.me/picGo/20210831000347.png?imgNote)

We can see that many of these hexagrams are a combination of "face the court" and "anti-war protest." This means that we should spend some effort on data cleaning to see if we can combine these synonyms into a clean token.



##  Topic modelling

### Use pyLDAvis for topic modeling exploration

Topic modeling is the process of using unsupervised learning techniques to extract the main topics that appear in the document set.

[Latent Dirichlet Allocation](https://towardsdatascience.com/light-on-math-machine-learning-intuitive-guide-to-latent-dirichlet-allocation-437c81220158?ref=hackernoon.com) (LDA) is an easy-to-use and efficient topic modeling model. Each document is represented by a topic distribution, and each topic is represented by a word distribution.

Once the documents are classified into topics, you can delve into the data for each topic or topic group.

But before entering topic modeling, we must do some preprocessing of the data. we will:

Tokenization: The process of converting sentences into tokens or word lists. remove stopwordslemmatize: Reduce the deformed form of each word to a common base or root. Convert to word bag: word bag is a dictionary where the key is the word (or ngram/tokens) and the value is the number of times each word appears in the corpus.

With NLTK, you can easily tokenize and formalize:

```python
import nltk
nltk.download('punkt')
nltk.download('wordnet')

# output
"""
[nltk_data] Downloading package punkt to /Users/xx/nltk_data...
[nltk_data]   Package punkt is already up-to-date!
[nltk_data] Downloading package wordnet to /Users/xx/nltk_data...
[nltk_data]   Unzipping corpora/wordnet.zip.
True
"""

def preprocess_news(df):
    corpus=[]
    stem=PorterStemmer()
    lem=WordNetLemmatizer()
    for news in df['headline_text']:
        words=[w for w in word_tokenize(news) if (w not in stop)]
        
        words=[lem.lemmatize(w) for w in words if len(w)>2]
        
        corpus.append(words)
    return corpus
  
corpus = preprocess_news(df)

# Now, let's use gensim to create a bag of words model
dic=gensim.corpora.Dictionary(corpus)
bow_corpus = [dic.doc2bow(doc) for doc in corpus]

# We can finally create the LDA model:
lda_model =  gensim.models.LdaMulticore(bow_corpus, 
                                   num_topics = 4, 
                                   id2word = dic,                                    
                                   passes = 10,
                                   workers = 2)

lda_model.show_topics()

# output
"""
[(0,
  '0.010*"say" + 0.007*"cup" + 0.006*"war" + 0.005*"world" + 0.005*"back" + 0.005*"plan" + 0.005*"green" + 0.004*"win" + 0.004*"woman" + 0.004*"new"'),
 (1,
  '0.010*"govt" + 0.009*"war" + 0.009*"new" + 0.007*"may" + 0.005*"sars" + 0.005*"call" + 0.005*"protest" + 0.005*"boost" + 0.005*"group" + 0.004*"hospital"'),
 (2,
  '0.018*"police" + 0.015*"baghdad" + 0.014*"man" + 0.005*"missing" + 0.005*"claim" + 0.005*"court" + 0.005*"australia" + 0.004*"move" + 0.004*"murder" + 0.004*"charged"'),
 (3,
  '0.030*"iraq" + 0.015*"war" + 0.007*"iraqi" + 0.007*"council" + 0.006*"troop" + 0.005*"killed" + 0.004*"crash" + 0.004*"soldier" + 0.004*"open" + 0.004*"say"')]
"""
```

Theme 0 represents things related to the Iraq war and the police. Theme 3 shows Australia's involvement in the Iraq War.

You can print all the topics and try to understand them, but there are tools that can help you run this data exploration more effectively. pyLDAvis is such a tool, it can interactively visualize the results of LDA.



### Visualize the topics

```python
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim_models.prepare(lda_model, bow_corpus, dic)
vis
```

![image-20210831000602995](http://qiniu.hivan.me/picGo/20210831000603.png?imgNote)

On the left, the area of each circle represents the importance of the topic relative to the corpus. Because there are four themes, we have four circles.

The distance between the center of the circle indicates the similarity between themes. Here you can see that Topic 3 and Topic 4 overlap, which indicates that the themes are more similar. On the right, the histogram of each topic shows the top 30 related words. For example, in topic 1, the most relevant words are "police", "new", "may", "war", etc.

Therefore, in our case, we can see many war-related words and topics in the news headlines.



### Wordclouds

Wordcloud is a great way to represent text data. The size and color of each word appearing in the word cloud indicate its frequency or importance.

It is easy to create a [wordcloud](https://amueller.github.io/word_cloud/index.html?ref=hackernoon.com) [using python](https://amueller.github.io/word_cloud/index.html?ref=hackernoon.com), but we need to provide data in the form of a corpus.

```python
stopwords = set(STOPWORDS)

def show_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        max_words=100,
        max_font_size=30, 
        scale=3,
        random_state=1 
        )
    
    wordcloud=wordcloud.generate(str(data))

    fig = plt.figure(1, figsize=(12, 12))
    plt.axis('off')
 
    plt.imshow(wordcloud)
    plt.show()
    
show_wordcloud(corpus)
```

![image-20210831000635924](http://qiniu.hivan.me/picGo/20210831000636.png?imgNote)

Similarly, you can see that terms related to war are highlighted, indicating that these words often appear in news headlines.

There are many parameters that can be adjusted. Some of the most famous are:

stopwords: stop a group of words appearing in the image. max_words: Indicates the maximum number of words to be displayed. max_font_size: Maximum font size.

There are many other options to create beautiful word clouds. For more detailed information, you can refer to here.

## Text sentiment

Sentiment analysis is a very common natural language processing task in which we determine whether the text is positive, negative or neutral. This is very useful for finding sentiments related to comments and comments, allowing us to gain some valuable insights from text data.

There are many projects that can help you use python for sentiment analysis. I personally like [TextBlob](https://github.com/sloria/TextBlob?ref=hackernoon.com) and [Vader Sentiment.](https://github.com/cjhutto/vaderSentiment?ref=hackernoon.com)

```python
from textblob import TextBlob
TextBlob('100 people killed in Iraq').sentiment

# output
"""
Sentiment(polarity=-0.2, subjectivity=0.0)
"""
```



### Textblob

Textblob is a python library built on top of nltk. It has been around for a while and is very easy to use.

The sentiment function of TextBlob returns two attributes:

Polarity: It is a floating-point number in the range of [-1,1], where 1 means a positive statement and -1 means a negative statement. Subjectivity: refers to how personal opinions and feelings affect someone’s judgment. The subjectivity is expressed as a floating point value with a range of [0,1].

I will run this feature on news headlines.

TextBlob claims that the text "100 people killed in Iraq" is negative, not a view or feeling, but a statement of fact. I think we can agree to TextBlob here.

Now that we know how to calculate these sentiment scores, we can use histograms to visualize them and explore the data further.

```python
def polarity(text):
    return TextBlob(text).sentiment.polarity

df['polarity_score']=df['headline_text'].\
   apply(lambda x : polarity(x))
df['polarity_score'].hist()
```

![image-20210831000735233](http://qiniu.hivan.me/picGo/20210831000735.png?imgNote)

You will see that the polarity is mainly between 0.00 and 0.20. This shows that most news headlines are neutral.

Let's categorize news as negative, positive, and neutral based on the scores for a more in-depth study.

### Postive , Negative or Neutral ?

```python

def sentiment(x):
    if x<0:
        return 'neg'
    elif x==0:
        return 'neu'
    else:
        return 'pos'
    
df['polarity']=df['polarity_score'].\
   map(lambda x: sentiment(x))
  
plt.bar(df.polarity.value_counts().index,
        df.polarity.value_counts())
```

![image-20210831000804842](http://qiniu.hivan.me/picGo/20210831000804.png?imgNote)

Yes, 70% of news is neutral, only 18% of positive news and 11% of negative news.

Let's look at the positive and negative headlines.

```python
df[df['polarity']=='neg']['headline_text'].head(5)

# output
"""
7     aussie qualifier stosur wastes four memphis match
23               carews freak goal leaves roma in ruins
28     council chief executive fails to secure position
34                   dargo fire threat expected to rise
40        direct anger at govt not soldiers crean urges
Name: headline_text, dtype: object
"""
```



### Vader

The next library we are going to discuss is VADER. Vader is better at detecting negative emotions. It is very useful in the context of social media text sentiment analysis.

The VADER or Valence Aware dictionary and sentiment reasoner is an open source sentiment analyzer pre-built library based on rules/dictionaries and is protected by the MIT license.

The VADER sentiment analysis class returns a dictionary that contains the possibility that the text appears positive, negative, and neutral. Then, we can filter and select the emotion with the highest probability.

We will use VADER to perform the same analysis and check if the difference is large.

```python
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()

def get_vader_score(sent):
    # Polarity score returns dictionary
    ss = sid.polarity_scores(sent)
    #return ss
    return np.argmax(list(ss.values())[:-1])
  
 
"""
[nltk_data] Downloading package vader_lexicon to
[nltk_data]     /Users/xx/nltk_data...
"""

df['polarity']=df['headline_text'].\
    map(lambda x: get_vader_score(x))
polarity=df['polarity'].replace({0:'neg',1:'neu',2:'pos'})

plt.bar(polarity.value_counts().index,
        polarity.value_counts())
```

![image-20210831000924225](http://qiniu.hivan.me/picGo/20210831000924.png?imgNote)

Yes, the distribution is slightly different. There are even more headlines classified as neutral 85%, and the number of negative news headlines has increased (to 13%).



## Named Entity Recognition

Named entity recognition is an information extraction method in which entities existing in the text are classified into predefined entity types, such as "person", "location", "organization" and so on. By using NER, we can gain insight into the entities that exist in a given text data set of entity types.

Let us consider an example of a news article.

In the above news, the named entity recognition model should be able to recognize
Entities, such as RBI as an organization, Mumbai and India as Places, etc.

There are three standard libraries for named entity recognition:

- [Stanford Nell](https://nlp.stanford.edu/software/CRF-NER.shtml?ref=hackernoon.com)
- [space](https://spacy.io/?ref=hackernoon.com)
- [NLTK](https://www.nltk.org/?ref=hackernoon.com)



**I will use spaCy**, which is an open source library for advanced natural language processing tasks. It is written in Cython and is known for its industrial applications. In addition to NER, **spaCy also provides many other functions, such as pos mark, word to vector conversion, etc. **

[SpaCy’s Named Entity Recognition](https://spacy.io/api/annotation?ref=hackernoon.com#section-named-entities) has been published in [OntoNotes 5](https://catalog.ldc.upenn.edu /LDC2013T19?ref=hackernoon.com) has been trained on the corpus and supports the following entity types



There are three kinds of [pre-trained models for English](https://spacy.io/models/en/?ref=hackernoon.com) in SpaCy. I will use *en_core_web_sm* to complete our task, but you can try other models.

To use it, we must first download it:

```python
# !python -m spacy download en_core_web_sm

# Now we can initialize the language model:

import spacy
from spacy import displacy
import en_core_web_sm

nlp = en_core_web_sm.load()

# nlp = spacy.load("en_core_web_sm")

# One of the advantages of Spacy is that we only need to apply the nlp function once, and the entire background pipeline will return the objects we need

doc=nlp('India and Iran have agreed to boost the economic \
viability of the strategic Chabahar port through various measures, \
including larger subsidies to merchant shipping firms using the facility, \
people familiar with the development said on Thursday.')

[(x.text,x.label_) for x in doc.ents]

 
"""
[('India', 'GPE'), ('Iran', 'GPE'), ('Chabahar', 'GPE'), ('Thursday', 'DATE')]
"""
```


We can see that India and Iran are confirmed as geographic locations (GPE), Chabahar is confirmed as a person, and Thursday is confirmed as a date.

We can also use the display module in spaCy to visualize the output.

```python
from spacy import displacy

displacy.render(doc, style='ent')
```





![image-20210831001008590](http://qiniu.hivan.me/picGo/20210831001008.png?imgNote)

This can make sentences with recognized entities look very neat, and each entity type is marked with a different color.

Now that we know how to perform NER, we can further explore the data by performing various visualizations on the named entities extracted from the data set.

First, we will run named entity recognition on news headlines and store entity types.

###  NER Analysis

```python
def ner(text):
    doc=nlp(text)
    return [X.label_ for X in doc.ents]
  
ent=df['headline_text'].apply(lambda x : ner(x))
ent=[x for sub in ent for x in sub]
counter=Counter(ent)
count=counter.most_common()

# Now, we can visualize the entity frequency:
x,y=map(list,zip(*count))
sns.barplot(x=y,y=x)
```

![image-20210831001044045](http://qiniu.hivan.me/picGo/20210831001044.png?imgNote)

Now we can see that GPE and ORG dominate the headlines, followed by the PERSON entity.

We can also visualize the most common tokens for each entity. Let's check which places appear the most in news headlines.

### Most common GPE

```python
def ner(text,ent="GPE"):
    doc=nlp(text)
    return [X.text for X in doc.ents if X.label_ == ent]
  
gpe=df['headline_text'].apply(lambda x: ner(x,"GPE"))
gpe=[i for x in gpe for i in x]
counter=Counter(gpe)

x,y=map(list,zip(*counter.most_common(10)))
sns.barplot(y,x)
```

![image-20210831001111535](http://qiniu.hivan.me/picGo/20210831001111.png?imgNote)

I think we can confirm the fact that "America" means America in news headlines. Let's also find the most common names that appear on news headlines.

### Most common person

```python
per=df['headline_text'].apply(lambda x: ner(x,"PERSON"))
per=[i for x in per for i in x]
counter=Counter(per)

x,y=map(list,zip(*counter.most_common(10)))
sns.barplot(y,x)
```

![image-20210831001135765](http://qiniu.hivan.me/picGo/20210831001135.png?imgNote)

Saddam Hussein and George Bush served as presidents of Iraq and the United States during the war. In addition, we can see that the model is far from perfect to classify "vic govt" or "nsw govt" as individuals rather than government agencies.



### Pos tagging

Use nltk for all parts of speech markup, but there are other libraries that can do the job well (spaacy, textblob).

```python
import nltk
nltk.download('averaged_perceptron_tagger')

sentence="The greatest comeback stories in 2019"
tokens=word_tokenize(sentence)
nltk.pos_tag(tokens)

# Notice:
# You can also use the spacy.displacy module to visualize the sentence part of the speech and its dependency graph.

doc = nlp('The greatest comeback stories in 2019')
displacy.render(doc, style='dep', jupyter=True, options={'distance': 90})
```

![image-20210831001212360](http://qiniu.hivan.me/picGo/20210831001212.png?imgNote)

We can observe various dependency labels here. For example, the DET tag indicates the relationship between the word "the" and the noun "stories".

You can check the list of dependency labels and their meanings [here](https://universaldependencies.org/u/dep/index.html?ref=hackernoon.com).

Okay, now that we know what a POS tag is, let's use it to explore the title data set.

### Analysing pos tags

```python
def pos(text):
    pos=nltk.pos_tag(word_tokenize(text))
    pos=list(map(list,zip(*pos)))[1]
    return pos
  
tags=df['headline_text'].apply(lambda x : pos(x))
tags=[x for l in tags for x in l]
counter=Counter(tags)
x,y=list(map(list,zip(*counter.most_common(7))))

sns.barplot(x=y,y=x)
```

![image-20210831001251251](http://qiniu.hivan.me/picGo/20210831001251.png?imgNote)

We can clearly see that nouns (NN) dominate in news headlines, followed by adjectives (JJ). This is typical for news reports, and for art forms, higher adjective (ADJ) frequencies may happen a lot.

You can investigate this in more depth by investigating the most common singular nouns in news headlines. Let us find out.

Nouns such as "war", "Iraq", and "person" dominate the news headlines. You can use the above functions to visualize and check other parts of the voice.

### Most common Nouns

```python
def get_adjs(text):
    adj=[]
    pos=nltk.pos_tag(word_tokenize(text))
    for word,tag in pos:
        if tag=='NN':
            adj.append(word)
    return adj


words=df['headline_text'].apply(lambda x : get_adjs(x))
words=[x for l in words for x in l]
counter=Counter(words)

x,y=list(map(list,zip(*counter.most_common(7))))
sns.barplot(x=y,y=x)
```



### Dependency graph

```python
doc = nlp('She sells seashells by  the seashore')
displacy.render(doc, style='dep', jupyter=True, options={'distance': 90})
```



## Text readability

### Textstat

```python
from textstat import flesch_reading_ease
df['headline_text'].apply(lambda x : flesch_reading_ease(x)).hist()
```



### complex headlines?

Almost all readability scores exceed 60. This means that an average of 11-year-old students can read and understand news headlines. Let's check all news headlines with a readability score below 5.

```python
x=[i for i in range(len(reading)) if reading[i]<5]

 
"""
rror loading preloads:
Failed to fetch dynamically imported module: https://file+.vscode-resource.vscode-webview.net/Users/xx/.vscode/extensions/ms-toolsai.jupyter-2021.8.1236758218/out/datascience-ui/errorRenderer/errorRenderer.js
"""

news.iloc[x]['headline_text'].head()

 
"""
Error loading preloads:
Failed to fetch dynamically imported module: https://file+.vscode-resource.vscode-webview.net/Users/xx/.vscode/extensions/ms-toolsai.jupyter-2021.8.1236758218/out/datascience-ui/errorRenderer/errorRenderer.js
"""
```

### Final thoughts

In this article, we discussed and implemented various exploratory data analysis methods for text data. Some are common and little known, but all of them can be an excellent addition to your data exploration toolkit.

Hope you will find some of them useful for your current and future projects.

To make data exploration easier, I created a "exploratory data analysis of natural language processing templates", which you can use for your work.

In addition, you may have seen that for each chart in this article, there is a code snippet to create it. Just click the button below the chart.

Happy exploring!

From: https://hackernoon.com/a-completeish-guide-to-python-tools-you-can-use-to-analyse-text-data-13g53wgr