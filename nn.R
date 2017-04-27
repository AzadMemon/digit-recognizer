# The neural network has 1 input layer, 1 output layer and 1 hidden layer.
# The neural network will process images of 28x28 px therefore our input layer will be 784 activation nodes + 1 bias unit
# The dimensions of the hidden layer are exactly the same as the input layer (i.e. 784 activation nodes + 1 bias unit)
# The output layer will classify the image into one of 10 categories (10 digits), therefore it will have 10 activation nodes

# TODO: Implement gradient descent manually
# TODO: Implement gradient checking
# TODO: Implement analysis 

# Read in testing data
testData <- read.csv("test.csv")
testDataX <- t(testData[,-1])
testDataY <- t(testData[,1])

# Read in training data
trainData <- read.csv("train.csv")
trainDataX <- t(trainData[,-1])
trainDataY <- t(trainData[,1])

costFunction <- function (predictedY, y, theta1, theta2, lambda) {
  # TODO: Change this so theta1 and theta2 will be unrolled, and have to be rebuilt
  m = nrow(predictedY)
  cost = 0;
  
  # Main component
  for (i in range(1, 10)) { # TODO: Fix this for loop, range isn't right
    yBinary <- 1*(y[,i]==i)
    cost = cost + colSums(rowSums((yBinary*log(predictedY) + (1-yBinary)*log(1-predictedY)))) 
  }
  
  # Regularized component
  cost = cost + lambda/(2*m) * (colSums(rowSums(theta1 * theta1)) + colSums(rowSums(theta2 * theta2)))
}

# Similar to gradient descent, R implements the sigmoid function as well
sigmoid <- function(z) {
  1/(1 + exp(-1*z)) 
}

forwardPropagation <- function(X, y, theta1, theta2) {
  # compute activation nodes for second layer
  XwithBias = cbind(1, X) # Add bias
  a2 = sigmoid(t(theta1) %*% XwithBias)
  
  # compute activation nodes for third layer
  a2WithBias = c(1, a2) # Add bias
  a3 = sigmoid(t(theta2) %*% a2WithBias)
  
  list(a2, a3)
}

backPropagation <- function(theta1, theta2, X, a2, a3, y, lambda) {
  num_examples = ncol(X)
  
  s3 = a3 - y
  s2 = (t(theta2) %*% s3) * a2 * (1 - a2);
  
  delta1 = s2 %*% t(a1)
  delta2 = s3 %*% t(a2)
  
  
  D1 = 1/num_examples * cbind(delta1[,1], delta1[,-1] + lambda*theta1[,-1])
  D2 = 1/num_examples * cbind(delta2[,1], delta2[,-1] + lambda*theta2[,-1])
  
  cbind(unlist(D1), unlist(D2))
}

# ----------------
# Orchestration
# ----------------

# Randomly initialize weights for each layer
# See the following: https://web.stanford.edu/class/ee373b/nninitialization.pdf

# Randomly initialize weights for layer 1
numActivationUnits1 <- 785 # Includes bias unit
numActivationUnits2 <- 784 # Does not include bias unit
epsilon1 = sqrt(6)/sqrt(numActivationUnits2 + numActivationUnits1)
theta1 <- 2*epsilon1*matrix(rnorm(numActivationUnits2*numActivationUnits1, mean=0, std=1), numActivationUnits2, numActivationUnits1) - epsilon1

# Randomly initialize weights for layer 2
numActivationUnits2 <- 785 # Includes bias unit
numActivationUnits3 <- 10 # Does not include bias unit
epsilon2 = sqrt(6)/sqrt(numActivationUnits3 + numActivationUnits2)
theta2 <- 2*epsilon2*matrix(rnorm(numActivationUnits3*numActivationUnits2, mean=0, std=1), numActivationUnits3, numActivationUnits2) - epsilon2

# Forward propagation
list[a2, a3] = forwardPropagation(trainDataX, trainDataY, theta1, theta2)

# TODO, this should be a function passed into grad descent
# Backpropagation
lambda = 5 # TODO: Update lambda value to be appropriate
list[d1, d2] = backPropagation(theta1, theta2, trainDataX, a2, a3, trainDataY, lambda)

# Gradient Descent
optim(cbind(unlist(theta1), unlist(theta2)), fn = costFunction, gr = backPropagation)

