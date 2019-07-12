# Introduction to Deep Learning
## Content
* The Perceptron
* Activation Function    
* Building Neural Nets with Perceptron
  * Simple Perceptron
  * Multi Output Perceptron
  * Single Layer Neural Network
  * Deep Neural Network
*Apply Neural Networks
*Quantifying Loss
*Empirical Loss
* Binary Cross Entropy Loss
* Mean Squared Error Loss
* Training Neural Networks
* Loss Optimization
* Gradient Descent
  * Computing Gradients
* Loss Function can be difficult to Optimize
* Setting the Learning Rate
  * How to deal with this?
* Adaptive Learning Rate Algorithm
  * Momentum
  * Adagrad
  * Adadelta
  * Adam
  * RMSProp
* Mini Batching
  * Stochastic Gradient Descent
  * Mini Batches while training
* Overfitting
* Regularization
  * Dropout
  * Early Stopping
## **Activation Function**
  Activation Functions are really very important for ANN to learn and make sense of some thing really complicated and non linear complex functions mapping between input and output response variables. They introduce non linearity to our neural networks If we don't apply activation function then the output signal would simply be a linear function. Activation functions should be differentiable 
### **Most Popular types of activation functions**
1.  Sigmoid or Logistic
2.  Tanh or Hyperbolic Tangent 
3.  Relu (Rectified Linear units)
### **1. Sigmoid Function:**
- Sigmoid Function takes a value as input and outputs another value between 0 and 1
- Its a non-linear and easy to work with when constructing a neural network model
- The good part about this function is that it is continuously differentiable over different values of z and has a 
      fixed output range

![Sigmoid_img](https://cdn-images-1.medium.com/max/1600/1*Xu7B5y9gp0iL5ooBj7LtWw.png)

**Math :**
    f(z) = \frac{1} {1 + e^{-z}}

### **Limitations of Sigmoid Function:**
- As z increases f(z) increases very slowly at some point it reaches 1. They are susceptible to vanishing gradient and models that uses sigmoid are slow learners
- When we have multiple hidden layers. The values that we get are of different magnitudes between 0 to 1(not zero centered) so it becomes hard to optimize  
- Sigmoid function is especially used for binary classification as part of output layer to capture the probability range from 0 to 1 Especially for multiple class softmax is used instead of sigmoid function

### **2. Tanh Function:**
- Tanh Function is a modified or scaled up function of sigmoid function. its values are dounded from -1 to 1
- We are able to get values of different signs which helps us in establishing which score to consider and which ones to ignore

![Tanh_img](https://cdn-images-1.medium.com/max/1600/1*1It8846pzYayiC0G_7FIBA.png)
**Math:**
f(z) = \frac{2}{1 + e^{-2z}}

### **Limitations:**
- The model slows down exponentially beyond the range -2 to 2 
- This function still has the vanishing gradient problem
### **Comparison of Sigmoid and Tanh:**
![Comparison](https://cdn-images-1.medium.com/max/1000/1*PHO7KDb_7nWPJS7x6wgzWQ.png)
- Tanh function is steeper when compared to sigmoid function. Our choose of using tanh and sigmoid completely depends on the requirement of gradient for the problem statement  
### **3. Relu**
- This function simply outputs 0 if it receives any input which is less than zero and for any positive value it returns the same z value like linear functions
![Relu_img](https://miro.medium.com/max/714/1*oePAhrm74RNnNEolprmTaQ.png)
**Math:**
f(z) = max(0,z)
### **Advantages:**
- Relu function has a derivative of 0 over half of its range(-ve values) and for positive input the derivative is 1. So vanishing gradient problem is vanished
- At a time only few neurons are activated making network sparse and efficient(sparsity is not always good)
- It is computationally economic compared to sigmoid and tanh
### **Limitations:**
- This function suffers from dying Relu problem. For activation's correspondent to values Z<0 the gradient will be zero because of which weights will not get adjusted. which means such neurans will stop responding to variation in error
- It is used in between input and output layers more specifically with in hidden layers
### **4. Leaky Relu:**
- It is a modified version of Relu to fix the problem of dying neurons. It introduce a small slope to keep the updates alive
- To keep it brief we take z<0 values which is y = 0 line and convert it into non-horizontal line by adding a small non zero gradient for z<0, f(z) = az
![Leaky_relu](https://cdn-images-1.medium.com/max/1600/1*A_Bzn0CjUgOXtPCJKnKLqA.jpeg)  
### **5. Maxout**
- This is another variety made from Relu and Leaky Relu 
![Relu_comp](https://cdn-images-1.medium.com/max/1600/0*qtfLu9rmtNullrVC.png)
### **Refered Links**
[Blog 1](https://towardsdatascience.com/activation-functions-in-neural-networks-83ff7f46a6bd0)
[Blog 2](https://towardsdatascience.com/activation-functions-and-its-types-which-is-better-a9a5310cc8f)
[Blog 3](https://missinglink.ai/guides/neural-network-concepts/7-types-neural-network-activation-functions-right/)
[Blog 4](https://medium.com/the-theory-of-everything/understanding-activation-functions-in-neural-networks-9491262884e0)
[Blog 5](https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6)

# **Loss Functions **
-Loss Function is used to measure the inconsistency between the predicted value and the actual value 
1. Mean Squared Error

## **Mean Squared Error:**
- MSE is used in linear regression as the performance measure 
- The method of minimizing MSE is called Ordinary least squares. OLS is that the optimized fitting line should be a line which minimizes the sum of distance of each point to the regression line
- If using sigmoid as the activation function the quadratic loss function suffers the problem of low convergence(learning rate) for other activation functions it won't have much effect
### Bias: 
It is the average difference between the estimator and true value. So its values can be positive when our model predicts more than the actual value, negative if predicted value is less than the actual value and zero if both the predicted and true value are same
                 bias = avg(Y(pred) - Y(actual))
### Variance:
It is the difference between Expectation of a squared random variable and the expectation of random variable squared 
  
[link 1](https://www.countbayesie.com/blog/2019/1/30/a-deeper-look-at-mean-squared-error)
 
## **Optimizer:**
Finding the best numerical solution to a given problem is an very important in machine learning . Optimizers combined with loss functions are the key factors that enable machine learning to work for your data
During the training process we tweak and change the parameters of our modelto try and minimize the loss function and make a prediction as correct as possible but how exactly do we do that? How do you change the parameters of your model, by how much and when
Optimizers tie together the loss function and model parameters by updating the model in response to the output of the loss function. Loss function is the guide to the optimizer telling it whether it is moving in right or wrong direction 
### **Gradient Descent:**
How it works:
* Calculate what a small change change in individual weight would do to the loss function (it like deciding which way to go)
* Adjust each individual weight based on its gradient(take a small step in the determined direction)
* Keep doing step 1 and 2 until the loss function gets as low as possible <br><br>
The tricky part of algorithm is understanding gradient they are partial derivatives and are a measure of change. They connect the loss function and weight <br>
One hiccup that you might experience during optimization is getting stuck at local minima when dealing with high dimensional data sets its possible you will find an area where you find an area where it seems like you have reached the lowest possible value for your loss function but it's just a local minima <br>
In order to avoid getting stuck in local minina we make sure we use proper learning rate 
### Learning Rate:
   

