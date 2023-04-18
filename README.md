# Discontinuity Capturing Shallow Neural Network (DCSNN)

This repository contains the code for the paper:

* [A Discontinuity Capturing Shallow Neural Network(DCSNN) for Elliptic Interface Problems](https://arxiv.org/abs/2106.05587)

In this paper, a new Discontinuity Capturing Shallow Neural Network (DCSNN) for approximating d-dimensional piecewise continuous functions and for solving elliptic interface problems is developed. There are three novel features in the present network; namely, (i) jump discontinuities are accurately captured, (ii) it is completely shallow, comprising only one hidden layer, (iii) it is completely mesh-free for solving partial differential equations. The crucial idea here is that a d-dimensional piecewise continuous function can be extended to a continuous function defined in (d+1)-dimensional space, where the augmented coordinate variable labels the pieces of each sub-domain. We then construct a shallow neural network to express this new function. Since only one hidden layer is employed, the number of training parameters (weights and biases) scales linearly with the dimension and the neurons used in the hidden layer. For solving elliptic interface problems, the network is trained by minimizing the mean square error loss that consists of the residual of the governing equation, boundary condition, and the interface jump conditions. We perform a series of numerical tests to demonstrate the accuracy of the present network. Our DCSNN model is efficient due to only a moderate number of parameters needed to be trained (a few hundred parameters used throughout all numerical examples), and the results indicate good accuracy. Compared with the results obtained by the traditional grid-based immersed interface method (IIM), which is designed particularly for elliptic interface problems, our network model shows a better accuracy than IIM. We conclude by solving a six-dimensional problem to demonstrate the capability of the present network for high-dimensional applications.

## Dependencies

* Matlab
* Python
	* [PyTorch](https://pytorch.org)
	* [Functorch](https://github.com/pytorch/functorch)

## Citations

```
@article{HLL22,
  author = {W.-F. Hu and T.-S. Lin and M.-C. Lai},
  doi = {10.1016/j.jcp.2022.111576},
  journal = {Journal of Computational Physics},
  number = {469},
  pages = {111576},
  title = {{A Discontinuity Capturing Shallow Neural Network for Elliptic Interface Problems}},
  year = {2022}
}