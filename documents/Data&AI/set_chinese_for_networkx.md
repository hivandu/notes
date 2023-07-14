# How to set up networkx in Chinese

## Problem description

Hi, everynone,

when we use networkx to display Chinese, we will find that Chinese cannot be displayed.

## Solution

1. Download the font in the attachment;

    > https://github.com/hivandu/practise/blob/master/resource/SimHei.ttf
2. Execute in jupyter notebook

```python
import matplotlib
print(matplotlib.__path__)
```


Find the path to matplotlib, and then cd to this path, after cd to this path, continue cd, cd to the path `map-data/fonts/ttf`. Then replace the file `DejaVuSans,ttf` with the file we just.

```bash
$ mv SimHei.ttf nx.draw(city_graph, city_location, with_labels = True, node_size = 10).ttf
```

Among them, the ttf font used. I have uploaded it to everyone.

