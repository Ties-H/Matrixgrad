# Matrixgrad
This is my implementation of a deep learning engine written from scratch in python, for now it is a reverse-mode autodiff framework for matrices that supports MLP's if the user correctly sets the matrix shapes. It expands on the idea of micrograd from Andrej Karpathy by introducing matrices instead of scalar values. The goal is to have it eventually work like an actual deep learning library and support operations such as convolution/batchnorm/attention etc, all while only using numpy and pure python.

## How to use:
At the core of this library is the Matrix class, you can initialize an M x N matrix where M and N can be arbitrarily large, on the matrices you initialize you can perform the operations from the matrix functions (matmul, matsub etc). When you are done with your computations/forward pass, you can call matrix.backprop() on the final matrix to run the backwards pass and compute the gradients. This allows you to create MLP's where backpropagation is handled automatically and you only need to write out the forward pass.

### Example use:
Here we create a MLP with a hidden layer with 64 neurons and train it using stochastic gradient descent.

weights1 = Matrix(np.random.randn(784,64)) #initialize weights
weights2 = Matrix(np.random.randn(64,10))

epochs = 500
lr = 0.1
for i in range(epochs):
    preactivation = X.matmul(weights1)
    h1 = preactivation.matrelu()
    preactivation2 = h1.matmul(weights2)
    loss, probs = preactivation2.softmax_cross_entropy(y)
    loss.backprop(probs)
    weights1.data -= lr * weights1.grad
    if i>epochs/2:
        lr = 0.01
    if i%100==0:
        print(loss.data)
 
print(loss.data)
