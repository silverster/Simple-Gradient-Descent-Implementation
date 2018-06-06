#Finding line of best fit using gradient descent
#import dependencies
import pandas as pd
import matplotlib.pyplot as plt

def calcError(Y,y):
	error = sum((Y-y)**2)/len(Y)
	return error 

def gradientDescent(X,Y,m,b,alpha,iters):
	N = len(X)
	for i in range(iters):
		m_grad = 0
		b_grad = 0
		y = m*X + b
		m_grad += -(2/N)*sum(X*(Y-(y)))
		b_grad += -(2/N)*sum(Y-(y))
		m = m - (alpha*m_grad)
		b = b - (alpha*b_grad)
	return m,b

#import dataset
dataset = pd.read_csv('data.csv')
X = dataset['Hours_studied']
Y = dataset['Marks_obtained']

##hyperparameters
#learning rate alpha
alpha = 0.0001   
m = 0 #slope initial value = 0
b = 0 #intercept initial value = 0
iters = 1000 #number of iterations

#Estimating Initial Error
y = m*X + b
print("Initial Error =",calcError(Y,y))


plt.scatter(X,Y)
plt.xlabel('Hours')
plt.ylabel('Marks')
plt.plot(X,y,'r',label='initial')
#plt.show()

#performing gradient descent to find best fit line
[m,b] = gradientDescent(X,Y,m,b,alpha,iters)
y = m*X + b
print("Final Error =",calcError(Y,y))

plt.plot(X,y,'g',label='final')
plt.legend()
plt.show()