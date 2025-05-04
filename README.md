# Fully Connected Neural Network from Scratch

> **Disclaimer**  
> This project was developed as part of my personal learning. While I may have consulted various educational resources (such as tutorials, documentation, blog posts, or videos) during its creation, I do not recall all specific sources. All code has been independently written and reflects my own understanding of the topic unless explicitly stated. Any resemblance to existing material is unintentional or stems from common practices in the field.


> Acknowledgments:  
> - [3Blue1Brown’s Deep Learning Playlist](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)  
> - [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) by Michael Nielsen
> - Finally, I would also like to acknowledge that ChatGPT was used to help teach concepts and used to help debug code. When used it was used in a teaching capacity.

## Overview

This project demonstrates the construction and training of a simple two-layer fully connected feedforward neural network built from scratch using only NumPy. It is trained on the MNIST handwritten digits dataset without using high-level deep learning libraries like TensorFlow/Keras for the model implementation.

The goal of this project is to deepen understanding of how backpropagation and gradient descent work under the hood by manually implementing each component of the training loop, forward pass, and backward pass.

## Key Features

- Two-layer fully connected neural network (784 → 16 → 10)
- Sigmoid activation function
- Manual backpropagation and weight updates
- Mean Squared Error (MSE) loss function with numerical gradient
- MNIST dataset input handling and one-hot label encoding
- Epoch-based training with performance printed per epoch
- Final evaluation on the test set

## Project Structure

- `FullyConnected.py`: Main script containing all class definitions, data preparation, network training, and testing

## Technologies Used

| Tool       | Purpose                            |
|------------|------------------------------------|
| Python     | Base programming language          |
| NumPy      | Matrix operations and computations |
| TensorFlow | MNIST data loading and processing  |

## Dataset

The model is trained and evaluated using the [MNIST dataset](http://yann.lecun.com/exdb/mnist/), which is loaded through TensorFlow's dataset API.

- Training set: 60,000 examples
- Test set: 10,000 examples
- Each input is a 28×28 grayscale image, flattened into a 784-length vector
- Labels are one-hot encoded for use in the network

## Setup Instructions

To run the project, follow these steps:

```bash
# 1. Clone the repository or download the script
git clone https://github.com/jhughes266/FullyConnected.git
cd FullyConnected

# 2. Install required dependencies
pip install numpy tensorflow

# 3. Run the training and evaluation script
python FullyConnected.py
```

## Output

The training loop displays the loss and accuracy per epoch, and the test set accuracy is printed at the end:

```
Epoch progress:  99.83 % Epoch:  1  Loss:  0.9162  Proportion Correct:  0.25
...
Test Complete! Correct Percentage: 91.35
Execution Finished
```

## My Role / What I Learned

- Built a two-layer neural network architecture from scratch using NumPy
- Learned how forward and backward passes are implemented without using deep learning libraries
- Improved understanding of gradient descent, loss computation, and parameter updates
- Gained confidence in working directly with raw MNIST data and custom training loops

## License

MIT License

Copyright (c) 2025 Jotham Hughes

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---

> This README was written with the assistance of OpenAI's ChatGPT.
