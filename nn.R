# The neural network has 1 input layer, 1 output layer and 1 hidden layer.
# The neural network will process images of 28x28 px therefore our input layer will be 784 activation nodes + 1 bias unit
# The dimensions of the hidden layer are exactly the same as the input layer (i.e. 784 activation nodes + 1 bias unit)
# The output layer will classify the image into one of 10 categories (10 digits), therefore it will have 10 activation nodes

# TODO: Implement gradient descent manually
# TODO: Implement gradient checking
# TODO: Implement analysis 

# Read in testing data
#testData <- read.csv("test.csv")
#testDataX <- t(testData[,-1])
#testDataY <- t(testData[,1])

# Read in training data
trainData <- read.csv("train.csv")
trainDataX <- t(trainData[1:5,-1])
trainDataY <- t(trainData[1:5,1])

fr <- function(X, Y, lambda, par) {
  # reroll parmeters
  theta1 <- matrix(head(par,615440), 784)
  theta2 <- matrix(tail(par,-615440), 10)
  
  output <- forwardPropagation(X, theta1, theta2)
  a2 = output$a2
  a3 = output$a3
  
  
  predictedY <- a3 # Output layer is the predicted values
  numTrainingExamples <- ncol(predictedY)
  # Compute the cost function
  cost <- 0;
  
  for (i in c(0:9)) {
    yBinary <- 1*(Y==i)
    cost <- cost + sum(rowSums(sweep(log(predictedY), MARGIN=2, yBinary, `*`) + sweep(log(1-predictedY), MARGIN=2, (1-yBinary), `*`)))
  }
  
  # Regularized component
  cost <- cost + lambda/(2*numTrainingExamples) * (sum(rowSums(theta1 * theta1)) + sum(rowSums(theta2 * theta2)))
  return(cost)
}

# Similar to gradient descent, R implements the sigmoid function as well
sigmoid <- function(z) {
  return(1/(1 + exp(-1*z)))
}

forwardPropagation <- function(X, theta1, theta2) {
  # compute activation nodes for second layer
  XwithBias <- rbind(1, X) # Add bias
  a2 <- sigmoid(theta1 %*% XwithBias)
  
  # compute activation nodes for third layer
  a2WithBias <- rbind(1, a2) # Add bias
  a3 <- sigmoid(theta2 %*% a2WithBias)
  
  return(list("a2" = a2, "a3" = a3))
}

grr <- function(X, Y, lambda, par) {
  # reroll parmeters
  theta1 <- matrix(head(par,615440), 784)
  theta2 <- matrix(tail(par,-615440), 10)
  
  output <- forwardPropagation(X, theta1, theta2)
  a2 = output$a2
  a3 = output$a3
  
  numExamples <- ncol(X)
  
  s3 <- sweep(a3, MARGIN=2, Y, `-`)
  s2 <- (t(theta2[,-1]) %*% s3) * a2 * (1 - a2)
  
  delta1 <- s2 %*% t(rbind(1, X))
  delta2 <- s3 %*% t(rbind(1, a2))
  
  
  D1 <- 1/numExamples * cbind(delta1[,1], delta1[,-1] + lambda*theta1[,-1])
  D2 <- 1/numExamples * cbind(delta2[,1], delta2[,-1] + lambda*theta2[,-1])
  
  return(c(as.vector(D1), as.vector(D2)))
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
theta1 <- 2*epsilon1*matrix(rnorm(numActivationUnits2*numActivationUnits1, mean=0), numActivationUnits2, numActivationUnits1) - epsilon1

# Randomly initialize weights for layer 2
numActivationUnits2 <- 785 # Includes bias unit
numActivationUnits3 <- 10 # Does not include bias unit
epsilon2 = sqrt(6)/sqrt(numActivationUnits3 + numActivationUnits2)
theta2 <- 2*epsilon2*matrix(rnorm(numActivationUnits3*numActivationUnits2, mean=0), numActivationUnits3, numActivationUnits2) - epsilon2

# Gradient Descent
lambda <- 3
init_theta <- c(as.vector(theta1), as.vector(theta2)) # Unrolled theta parameters
result <- optim(par = init_theta, fr, grr, X = trainDataX, Y = trainDataY, lambda = lambda, method="BFGS", control=list("maxit"=25, "ndeps"=0.1, "reltol"=0.0001))

print(result)
