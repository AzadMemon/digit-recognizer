# The neural network has 1 input layer, 1 output layer and 1 hidden layer.
# The neural network will process images of 28x28 px therefore our input layer will be 784 activation nodes + 1 bias unit
# The dimensions of the hidden layer are exactly the same as the input layer (i.e. 784 activation nodes + 1 bias unit)
# The output layer will classify the image into one of 10 categories (10 digits), therefore it will have 10 activation nodes

# Read in testing data
testData <- read.csv("test.csv")
testDataX <- testData[,-1]
testDataY <- testData[,1]

# Read in training data
trainData <- read.csv("train.csv")
trainDataX <- trainData[,-1]
trainDataY <- trainData[,1]



costFunction <- function () {
  
}

# Even though R offers libraries to comput gradient descent
# I thought it might offer a more comprehensive understanding if I implemented it myself
gradientDescent <- function() {
  
}

# Gradient checking ensures back propagation algorithm is implemented correctly
gradientChecking <- function () {
  
}

# Similar to gradient descent, R implements the sigmoid function as well
sigmoid <- function(z) {
  1/(1 + exp(-1*z)) 
}

forwardPropagation <- function(X, y, theta1, theta2) {
  # compute activation nodes for second layer
  XwithBias = cbind(1, X)
  a2 = sigmoid(XwithBias %*% t(theta1))
  
  # compute activation nodes for third layer
  a2WithBias = c(1, a2)
  a3 = sigmoid(a2WithBias %*% t(theta2))
  
  list(a2, a3)
}

backPropagation <- function() {
  
}

# ----------------
# Orchestration
# ----------------

# Randomly initialize weights for each layer
# See the following: https://web.stanford.edu/class/ee373b/nninitialization.pdf

# Randomly initialize weights for layer 1
numInputNodes1 <- 784
numOutputNodes1 <- 784
epsilon1 = sqrt(6)/sqrt(numOutputNodes1 + numInputNodes1)
theta1 <- 2*epsilon1*matrix(rnorm(numOutputNodes1*(numInputNodes1 + 1), mean=0, std=1), numOutputNodes1, numInputNodes1) - epsilon1

# Randomly initialize weights for layer 2
numInputNodes2 <- 784
numOutputNodes2 <- 10
epsilon2 = sqrt(6)/sqrt(numOutputNodes2 + numInputNodes2)
theta2 <- 2*epsilon2*matrix(rnorm(numOutputNodes2*(numInputNodes2 + 1), mean=0, std=1), numOutputNodes2, numInputNodes2) - epsilon2

# Forward propagation
for (i in range(1, nrow(trainDataX))) {
  
}
list[a2, a3] = forwardPropagation(trainDataX, trainDataY, theta1, theta2)

# 

# -------------------------
# Classification accuracy
# -------------------------



