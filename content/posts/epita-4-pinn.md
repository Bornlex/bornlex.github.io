+++
title = 'EPITA Courses - Continuous Physics Informed Neural Networks'
date = 2025-03-16T15:54:04+01:00
draft = false
+++


# Introduction

Neural networks require large amounts of data to converge. Those data need to represent the task the neural network is trying to learn.

Data collection is a tedious process, especially when collecting data can be difficult or expensive. In science, physics for instance, many phenomenon are described using theories that we know are working very well.

Using those data as regularization can help neural networks generalize better with less data.

In some area, like fluid dynamics, using statistical models can speed up simulations and avoid using computationally expensive system of equations (Navier Stokes for instance).

When trying to predict the behavior of a physical system, we most of the time want to know the state of the system at a certain point in time $t$.
We are looking an equation that looks like this:

$$
x(t) = f(t; \theta)
$$

Here $x$ is the true phisical state of the system at time $t$ and $f$ is our neural network that will approximate it. It is parametrize by $\theta$, the weights.

Of course, we will use some data that we collected in order to fit our model. This will look like a typical training, with a loss:

$$
L(\hat{Y}, Y) = \frac{1}{N} \sum^N_i(\hat{y}_i, y_i)^2
$$

Where:

- $\hat{y}_i \in \hat{Y}$ is a prediction made by the model
- $y \in Y$ is the set of ground truth

Of course, this is equivalent to:

$$
L(\hat{Y}, Y) = \frac{1}{N} \sum^N_i(f(t_i; \theta), y_i)^2
$$

But, because we want our model to follow the dynamics of a real physical system, we can insert into the loss a regularization term that corresponds to a physical equation.

Let us give a concrete example of a regularization term that corresponds to a real physical equation.

# Spring-mass System

## Contexte

Let us imagine the following situation: we have a spring-mass system on a table and we want a model that will predict the position of the system at any point in time $t$.

We note the real trajectory of the system $t \mapsto x(t)$.

<aside>
💡 Note that we are indeed in a continuous time series framework here, $t$ can be any real value.

</aside>

Because finding the exact solution might be difficult analytically (for the spring-mass system, the solution is actually not so difficult, so we could reach it analytically, but for some other systems it might not be that easy), we want a model that will predict the state of the system. To do this, we choose a neural network.

![Spring-mass System](/epita/spring.png)

Our spring-mass system.

The grey line represents the position of the mass (the green square) over time. The green dot is the position at time $t$.

Here, the problem we have is that the data we collected are all located at the beginning of the phenomenon.
The risk here is that we get a decent approximation until we reach the last point in time that we measured and after that, the network might behave strangely.

![Data](/epita/spring-data.png)

The orange dots are the measures we collected.

Let us notice that the mass is dampened, which might be not so easy to teach our neural network to replicate.

This is why we will have to introduce to our neural networks some physical lessons.

## Differential equation

Let us do some good old physics here by applying the Newton law:

$$
\sum_i \overrightarrow{F}_i = m\overrightarrow{a}_G
$$

The forces that take place here are:

- the weight $\overrightarrow{P}$
- the spring opposing force $\overrightarrow{F}$, that depends on how much the spring is elongated (Hooke’s law)
- friction $\overrightarrow{f}$, depends on the speed of the spring-mass system
- vertical reaction of the table on the spring-mass system $\overrightarrow{R}$

The spring-mass system not being very heavy, the vertical reaction of the table and the weight are cancelling each other.

Moreover, we are interested in the horizontal position of the spring-mass system, so we can project our equation on the horizontal axis and ignore those two forces.

Our differential equation then becomes:

$$
F + f = m a_G
$$

Which then becomes:

$$
-kx(t) - \alpha \dot{x}(t) = m \ddot{x}(t)
$$

Then:

$$
m \ddot{x}(t) + \alpha \dot{x}(t) + kx(t) = 0
$$

And finally:

$$
\ddot{x}(t) + \frac{\alpha}{m} \dot{x}(t) + \frac{k}{m}x(t) = 0
$$

This is a second order differential equation on the position of the spring-mass system $x(t)$.

<aside>
💡 Notice that the first order derivative of the position appeared only because we are taking into account the friction, otherwise it would still be a second order differential equation, but a bit simpler without the second term.

</aside>

## Regularization

Now that we have the differential equation for our system, we can use it to train our model. The easy way to use it during training is the loss function.

We can see it as a regularization term. We need our model to predict correctly the measures and to respect the dynamics given by the differential equation.

Those two constraints are going to be enforced the same way: by computing the mean square error with the expected value.

Our loss function will look like this:

$$
L = \frac{1}{N} \sum^N_i (y_i - f(x_i) )^2 + \lambda \frac{1}{N} \sum^N_i [\ddot{f}(x_i) + \frac{\alpha}{m} \dot{f}(x_i) + \frac{k}{m} f(x_i)]^2
$$

In order to compute the first order and second order derivatives in the second term, you will have to use the **torch.autograd.grad** function of the pytorch library two times, one for the first order derivative, and a second time for the second order derivative.

The $\lambda$ parameter is a weighting parameter.

Here is the documentation:

[torch.autograd.grad — PyTorch 2.1 documentation](https://pytorch.org/docs/stable/generated/torch.autograd.grad.html)

The grad function takes several arguments:

1. the term to differentiate (can be the output of the neural network here)
2. the term to differentiate with respect to (can be the input of the neural network here)
3. the gradient with respect to each output, in our case, this is going to be a tensor full of 1 of the same size than the output itself
4. **create_graph** argument that we will have to set to **True** because we need to compute the second order derivative, and so we need to create the full computation graph when differentiating the first time

Be careful with the output format of the grad function, it returns a tuple and you will need only the first element.

# Practicum

In order to train our neural network, we need to generate some synthetic data and sample some out of it:

```python
def oscillator(d, w0, x):
    assert d < w0
    w = np.sqrt(w0 ** 2 - d ** 2)
    phi = np.arctan(-d / w)
    A = 1 / (2 * np.cos(phi))
    cos = torch.cos(phi + w * x)
    sin = torch.sin(phi + w * x)
    exp = torch.exp(-d * x)
    y  = exp * 2 * A * cos
    return y

d, w0 = 2, 20
mu, k = 2 * d, w0 ** 2. # the alpha/m and k/m constants in the reg term

# generate some synthetic data
x = torch.linspace(0, 1, 500).view(-1, 1)
y = oscillator(d, w0, x).view(-1, 1)

# sample some data to train on
x_data = x[0:200:20]
y_data = y[0:200:20]

# display the dataset and sample used for training
plt.figure()
plt.plot(x, y, label="Exact solution")
plt.scatter(x_data, y_data, color="tab:orange", label="Training data")
plt.legend()
plt.show()
```

## Display functions

Here is a function to display the results of your model:

```python
def plot_result(x, y, x_data, y_data, yh, xp=None):
    """
    x: the xs dataset
    y: the ys dataset
    x_data: the data used for prediction: t
    y_data: the ground truth x(t)
    yh: the prediction of the model: f(t)
    """
    plt.figure(figsize=(8, 4))
    plt.plot(x, y, color="grey", linewidth=2, alpha=0.8, label="Exact solution")
    plt.plot(x, yh, color="tab:blue", linewidth=4, alpha=0.8, label="Neural network prediction")
    plt.scatter(x_data, y_data, s=60, color="tab:orange", alpha=0.4, label='Training data')
    if xp is not None:
        plt.scatter(
            xp, -0 * torch.ones_like(xp), s=60, color="tab:green", alpha=0.4,
            label='Physics loss training locations'
        )
    l = plt.legend(loc=(1.01, 0.34), frameon=False, fontsize="large")
    plt.setp(l.get_texts(), color="k")
    plt.xlim(-0.05, 1.05)
    plt.ylim(-1.1, 1.1)
    plt.text(1.065, 0.7,"Training step: %i"%(i+1),fontsize="xx-large",color="k")
    plt.axis("off")
```

## Model

You will have to create a simple neural network using pytorch:

```python
class SimpleNN(nn.Module):
		"""
		3 layers:
			1. input to hidden
			2. hidden to hidden
			3. hidden to output
		1 activation function: tanh used at every layer but the last one.
		"""
    def __init__(self):
        super().__init__()
        # FIX ME

    def forward(self, x: torch.Tensor):
				# FIX ME
        pass
```

## Training

Most of the work (actually the regularization term) will be held inside the training function. You will need to use the autograd.grad function in order to compute the first order and the second order derivatives.

Make sure to read the documentation linked above and to include the term inside the loss function.

Training will include the following steps:

```python
1. Create the model
2. Create the optimizer

for i in range(1000):
		3. zero grad
		4. make predictions
		5. compute loss and regularization term if needed (only in the second model training)
		6. backward loss
		7. step optimizer
		8. every n step, display results
```

### Regular neural network

In order to compare the results with and without this regularization term, we will start by training a neural network without with only the MSE loss.

At the end of training with parameters:

- $lr = 1.10^{-3}$
- $epochs = 10^3$

The prediction should look like this:

![Spring Fit Wrong](/epita/spring-fit1.png)

### Physics informed neural network

Then we will train the same neural network but with the regularization term. Training will be done with parameters:

- $lr = 1.10^{-4}$
- $epochs = 2.10^4$

As you can see in the loss function above, there is a $\lambda$ parameter, we can use $1.10^{-4}$ at first, it should give good results.

The regularization term will not be computed on the same values than the MSE loss term otherwise it might perturbate the computation graph.

So you will generate 30 points between 0 and 1 using torch.linspace. Do not forget to call the require gradient method on them, otherwise gradient won’t be computed during training.

At the end of training, you should see something like this:

![Spring fit right](/epita/spring-fit2.png)

# Resources

Here is a very interesting paper that you might want to read. It is different from most papers where basically nothing new is created, this one introduces a concept that is the limit of an infinitely large resnet network.

[Neural Ordinary Differential Equations](https://arxiv.org/abs/1806.07366)

The implementation is actually short but not so easy to understand. Also, it uses autograd in an interesting way.

Also, here is an article very interesting about speeding up computation graphs, which is the core algorithmic part of deep learning.

[Adjoint State Method, Backpropagation and Neural ODEs | Ilya Schurov](https://ilya.schurov.com/post/adjoint-method/)

Finally, some more physical based neural networks that make use of physical conservation laws.

[Hamiltonian Neural Networks](https://greydanus.github.io/2019/05/15/hamiltonian-nns/)