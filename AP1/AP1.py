#Kholby Lawson, CS544, Fall 2015

# imports
import pandas as pd
from sklearn.linear_model import LinearRegression

'''
Table contains these columns:
    1. mpg:           continuous
    2. cylinders:     multi-valued discrete
    3. displacement:  continuous
    4. horsepower:    continuous
    5. weight:        continuous
    6. acceleration:  continuous
    7. model year:    multi-valued discrete
    8. origin:        multi-valued discrete
    9. car name:      string (unique for each instance)
'''

#read raw data
names=['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin','Car Name']
raw_data = pd.read_table('http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data', delim_whitespace=True, names=names, na_values=['?'])


# regression with all 4 features
features = ['Displacement', 'Horsepower', 'Weight', 'Acceleration']
data = raw_data.dropna()
X = data[features]
y = data.MPG
linreg = LinearRegression()
linreg.fit(X,y)
print 'Linear regression with displacement, horsepower, weight, and acceleration:\n'
print 'Coefficients:  ' + str(zip(features, linreg.coef_))
print 'Intercept:  ' + str(linreg.intercept_)
print 'R-squared: ' + str(linreg.score(X, y)) + '\n'


# regression with Displacement only
X = data[['Displacement']]
linreg.fit(X,y)
print 'Linear regression with displacement only:\n'
print 'Coefficient:  ' + str(linreg.coef_)
print 'Intercept:  ' + str(linreg.intercept_)
print 'R-squared:  ' + str(linreg.score(X,y)) + '\n'

# regression with Horsepower only
X = data[['Horsepower']]
linreg.fit(X,y)
print 'Linear regression with horsepower only:\n'
print 'Coefficient:  ' + str(linreg.coef_)
print 'Intercept:  ' + str(linreg.intercept_)
print 'R-squared:  ' + str(linreg.score(X,y)) + '\n'

# regression with Weight only
X = data[['Weight']]
linreg.fit(X,y)
print 'Linear regression with weight only:\n'
print 'Coefficient:  ' + str(linreg.coef_)
print 'Intercept:  ' + str(linreg.intercept_)
print 'R-squared:  ' + str(linreg.score(X,y)) + '\n'

# regression with Acceleration only
X = data[['Acceleration']]
linreg.fit(X,y)
print 'Linear regression with acceleration only:\n'
print 'Coefficient:  ' + str(linreg.coef_)
print 'Intercept:  ' + str(linreg.intercept_)
print 'R-squared:  ' + str(linreg.score(X,y)) + '\n'

#regression with displacement, horsepower, weight, acceleration, and cylinders
features = ['Displacement','Horsepower', 'Weight', 'Acceleration', 'Cylinders']
X = data[features]
linreg.fit(X,y)
print 'Linear regression with displacement, horsepower, weight, acceleration, and cylinders:\n'
print 'Coefficients: ' + str(zip(features, linreg.coef_))
print 'Intercept:  ' + str(linreg.intercept_)
print 'R-squared:  ' + str(linreg.score(X,y))

