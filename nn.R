  # The neural network has 1 input layer, 1 output layer and 1 hidden layer.
  # The neural network will process images of 28x28 px therefore our input layer will be 784 activation nodes + 1 bias unit
  # The dimensions of the hidden layer are exactly the same as the input layer (i.e. 784 activation nodes + 1 bias unit)
  # The output layer will classify the image into one of 10 categories (10 digits), therefore it will have 10 activation nodes
  
  # TODO: Implement gradient checking
  # TODO: Implement analysis 
  
  # Read in testing data
  #testData <- read.csv("test.csv")
  #testDataX <- t(testData[,-1])
  #testDataY <- t(testData[,1])

  # source("lbfgsb3_.R")

  # Read in training data
  trainData <- read.csv("train.csv")
  trainDataX <- t(head(trainData, 42000)[,-1])
  trainDataX <- trainDataX/255
  trainDataY <- t(head(trainData, 42000)[,1])
  
  library(sigmoid);
  
  forwardPropagation <- function(X, theta1, theta2) {
    # compute activation nodes for second layer
    XwithBias <- rbind(1, X) # Add bias
    a2 <- sigmoid(theta1 %*% XwithBias)
    
    # compute activation nodes for third layer
    a2WithBias <- rbind(1, a2) # Add bias
    a3 <- sigmoid(theta2 %*% a2WithBias)
    
    return(list("a2" = a2, "a3" = a3))
  }
  
  costFunction <- function(X, Y, lambda) {
    function(par) {
      # reroll parmeters
      theta1 <- matrix(head(par,615440), 784)
      theta2 <- matrix(tail(par,-615440), 10)
      
      output <- forwardPropagation(X, theta1, theta2)
      a2 <- output$a2
      a3 <- output$a3
      
      
      predictedY <- a3 # Output layer is the predicted values
      numTrainingExamples <- ncol(predictedY)
      
      # Compute the cost function
      cost <- -1*sum(rowSums(log(predictedY)*Y + log(1-predictedY)*(1-Y))) + lambda/(2*numTrainingExamples) * (sum(rowSums(theta1 %*% t(theta1))) + sum(rowSums(theta2 %*% t(theta2))))

      return(cost)
    }
  }
  
  gradFunction <- function(X, Y, lambda) {
    function(par) {
      # reroll parmeters
      theta1 <- matrix(head(par,615440), 784)
      theta2 <- matrix(tail(par,-615440), 10)
      
      output <- forwardPropagation(X, theta1, theta2)
      a2 <- output$a2
      a3 <- output$a3
      
      numExamples <- ncol(X)
      s3 <- a3 - Y
      s2 <- (t(theta2[,-1]) %*% s3) * a2 * (1 - a2)
      
      delta1 <- s2 %*% t(rbind(1, X))
      delta2 <- s3 %*% t(rbind(1, a2))
      
      D1 <- 1/numExamples * cbind(delta1[,1], delta1[,-1] + lambda*theta1[,-1])
      D2 <- 1/numExamples * cbind(delta2[,1], delta2[,-1] + lambda*theta2[,-1])
      
    
      return(c(as.vector(D1), as.vector(D2)))
    }
  }
  computeNumericalGradient <- function(J, theta) {
    #COMPUTENUMERICALGRADIENT Computes the gradient using "finite differences"
    #and gives us a numerical estimate of the gradient.
    #   numgrad <- COMPUTENUMERICALGRADIENT(J, theta) computes the numerical
    #   gradient of the  J  around theta. Calling y <- function(theta) should 
    #   return the function value at theta.
    
    # Notes: The following code implements numerical gradient checking, and
    #        returns the numerical gradient.It sets numgrad[i] to (a numerical
    #        approximation of) the partial derivative of J with respect to the
    #        i-th input argument, evaluated at theta. (i.e., numgrad[i] should
    #        be the (approximately) the partial derivative of J with respect
    #        to theta[i].)
    #
    
    numgrad <- rep(0,length(theta))
    perturb <- rep(0,length(theta))
    e <- 1e-4
    for (p in 1:length(theta)) {
      # Set perturbation vector
      perturb[p] <- e
      loss1 <- J(theta - perturb)
      loss2 <- J(theta + perturb)
      # Compute Numerical Gradient
      numgrad[p] <- (loss2 - loss1) / (2 * e)
      perturb[p] <- 0
    }
    
    numgrad
  }
  
  # ----------------
  # Orchestration
  # ----------------
  
  # Randomly initialize weights for each layer
  # See the following: https://web.stanford.edu/class/ee373b/nninitialization.pdf
  
  # Randomly initialize weights for layer 1
  numActivationUnits1 <- 785 # Includes bias unit
  numActivationUnits2 <- 784 # Does not include bias unit
  epsilon1 <- sqrt(6)/sqrt(numActivationUnits2 + numActivationUnits1)
  theta1 <- 2*epsilon1*matrix(runif(numActivationUnits2*numActivationUnits1, min=0, max=1), numActivationUnits2, numActivationUnits1) - epsilon1
  
  # Randomly initialize weights for layer 2
  numActivationUnits2 <- 785 # Includes bias unit
  numActivationUnits3 <- 10 # Does not include bias unit
  epsilon2 <- sqrt(6)/sqrt(numActivationUnits3 + numActivationUnits2)
  theta2 <- 2*epsilon2*matrix(runif(numActivationUnits3*numActivationUnits2, min=0, max=1), numActivationUnits3, numActivationUnits2) - epsilon2
  
  # Gradient Descent
  lambda <- 5
  init_theta <- c(as.vector(theta1), as.vector(theta2)) # Unrolled theta parameters
  
  trainDataYmodified <- matrix(0, 10, length(trainDataY))
  for (i in c(1:length(trainDataY))) {
    for (j in c(0:9)) {
      trainDataYmodified[j + 1, i] = 1*(trainDataY[i]==j)
    }
  }
  
  fn <- costFunction(trainDataX, trainDataYmodified, lambda)
  gr <- gradFunction(trainDataX, trainDataYmodified, lambda)
  
  #grad = gr(init_theta);
  #print(grad)
  #numgrad = computeNumericalGradient(fn, init_theta)
  #diff <- norm(as.matrix(numgrad - grad)) / norm(as.matrix(numgrad + grad))
  #print(diff)
  
  # After you have completed the assignment, change the maxit to a larger
  # value to see how more training helps.
  
  
  
  result <- optim(par = init_theta, fn, gr, method="L-BFGS-B", control=list(maxit=500, trace=4))
  
  # Test using testing data
  # Pass X, theta1 and theta2 to feedforward propagation
  # Get back a3
  # Get max probability of a3
  write.csv(result$par, file="theta.csv")
  
  print(result)
