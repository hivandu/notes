# Boston house price CART regression tree


## On the code



```python
# CART regression tree prediction
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score,mean_absolute_error, mean_squared_error
from sklearn.tree import DecisionTreeRegressor,export_graphviz
import graphviz

# Prepare data set
boston = load_boston()

# Explore data
print(boston.feature_names)

# Get feature set and price
features = boston.data
prices = boston.target


# Randomly extract 33% of the data as the test set, and the rest as the training set
train_features, test_features, train_price, test_price = train_test_split(features,prices,test_size=0.33)

# Create CART regression tree
dtr = DecisionTreeRegressor()

# Fitting and constructing CART regression tree
dtr.fit(train_features, train_price)

# Predict housing prices in the test set
predict_price = dtr.predict(test_features)

grap_data = export_graphviz(dtr, out_file=None)
graph = graphviz.Source(grap_data)

# Result evaluation of test set
print(f'Regression tree mean squared deviation:',mean_squared_error(test_price, predict_price))
print(f'Regression tree absolute value deviation mean:',mean_absolute_error(test_price, predict_price))

# Generate regression tree visualization
graph.render('Boston')
```



!> Before running this code, please ensure that the relevant dependencies have been installed;