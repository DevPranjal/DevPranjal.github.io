---
title: JAX. Decoding the name.
updated: 2021-11-26
---

> Update: This post was one of the [winning work](https://twitter.com/weights_biases/status/1467131585573097487) under #27DaysOfJAX organized by Weights and Biases 

#### Some background

[JAX]() was introduced in 2018 to provide a framework to get "easily programmable and highly performant ML system that targets CPUs, GPUs, and TPUs, capable of scaling to multi-core Cloud TPUs."

JAX might be just another machine learning framework, but its approach is very different from the others. It respects the functional programming paradigm, follows the NumPy API, introduces composable function transformations (the main reason JAX is easy to adopt), and uses a domain-specific JIT compiler.

Before getting started with JAX, let us familiarize ourselves with the topics of **J**IT compilation, **A**utograd, and the **X**LA compiler (yes, exactly, JAX)

<br>


#### Autograd

[Autograd](https://github.com/HIPS/autograd) is a project aimed at being able to differentiate native Python and Numpy code automatically. The [official repository](https://github.com/HIPS/autograd) states:
> It can handle a large subset of Python's features, including loops, ifs, recursion, and closures, and it can even take derivatives of derivatives. It supports reverse-mode differentiation (a.k.a. backpropagation), which means it can **efficiently take gradients of scalar-valued functions with respect to array-valued arguments**, as well as forward-mode differentiation, and the two can be composed arbitrarily. The main intended application of Autograd is gradient-based optimization.

##### Simple gradient descent with Autograd:

```python
from autograd import grad
import numpy as np


# defining model
real_weights = np.array([2.0, -3.4, 4.5])
# model = 2.0 * x1 - 3.4 * x2 + 4.5 * x3


# toy data
sample_data = np.random.normal(size=(100, 3))
sample_targets = np.matmul(sample_data, real_weights)


# defining loss
def loss(weights, data, target):
    predictions = np.matmul(data, weights)
    return np.sum(np.square(predictions - target))


# weight initialization
pred_weights = np.random.normal(size=(3,))
print(f"Loss before optimization: \
        {loss(pred_weights, sample_data, sample_targets)}")


################## Autograd is used here ##################
gradient_fn = grad(loss)

# gradient descent
for i in range(50):
    pred_weights -= 0.001 * gradient_fn(pred_weights,
                            sample_data, sample_targets)
    # gradient will be taken with respect to the 
    # first parameter -> pred_weights


print(f"Loss after optimization: \
        {loss(pred_weights, sample_data, sample_targets)}")
```

##### How it works?

Autograd works by building a computational graph of a function whose gradient we require. It records all the transformations the input undergoes as it passes through the function (try reading more on how it handles loops, ifs). Each transformation (or function), `f_raw` is wrapped using the [`primitive`](https://github.com/HIPS/autograd/blob/01eacff7a4f12e6f7aebde7c4cb4c1c2633f217d/autograd/tracer.py#L31-L51) function, which spits out another function `f_wrapped` so that its gradient can be specified and its invocation can be recorded. When going through the graph, the gradients of these primitives can be calculated (predefined in Autograd as a mapping between primitive and its gradient). Finally, while propagating through the graph, the chain rule is applied to each node.


##### Trace printing

```python
# https://github.com/HIPS/autograd/blob/master/examples/print_trace.py

import autograd.numpy as np
from autograd.tracer import trace, Node

class PrintNode(Node):
    def __init__(self, value, fun, args, kwargs,
                    parent_argnums, parents):
        self.varname_generator = parents[0].varname_generator
        self.varname = next(self.varname_generator)
        args_or_vars = list(args)
        for argnum, parent in zip(parent_argnums, parents):
            args_or_vars[argnum] = parent.varname
        print('{} = {}({}) = {}'.format(
            self.varname, fun.__name__, ','.join(
                map(str,args_or_vars)), value
            )
        )

    # defined in autograd.tracer.Node, added here for reference
    # @classmethod
    # def new_root(cls, *args, **kwargs):
    #     root = cls.__new__(cls)
    #     root.initialize_root(*args, **kwargs)
    #     return root

    def initialize_root(self, x):
        self.varname_generator = make_varname_generator()
        self.varname = next(self.varname_generator)
        print('{} = {}'.format(self.varname, x))

def make_varname_generator():
    for i in range(65, 91):
        yield chr(i)
    raise Exception("Ran out of alphabet!")

def print_trace(f, x):
    start_node = PrintNode.new_root(x)
    trace(start_node, f, x)
    print()

def avg(x, y):
    return (x + y) / 2
def fun(x):
    y = np.sin(x + x)
    return avg(y, y)

print_trace(fun, 1.23)

# A = 1.23
# B = add(A, A) = 2.46
# C = sin(B) = 0.6300306299958922
# D = add(C, C) = 1.2600612599917844
# E = true_divide(D, 2) = 0.6300306299958922

# Traces can be nested, so we can also trace through grad(fun)
from autograd import grad
print_trace(grad(fun), 1.0)

# A = 1.0
# B = add(A, A) = 2.0
# C = sin(B) = 0.9092974268256817
# D = add(C, C) = 1.8185948536513634
# E = true_divide(D, 2) = 0.9092974268256817
# F = cos(B) = -0.4161468365471424
# G = multiply(1.0, F) = -0.4161468365471424
# H = add(ArrayVSpace_{'shape': (), 'dtype': dtype('float64')},
      G, G) = -0.8322936730942848
```

Read more [here](https://github.com/HIPS/autograd/blob/master/docs/tutorial.md)


<br>


#### Just-in-Time (JIT) compilation and the XLA

##### Python: Behind the scenes

When we say "the Python language", it's a language specification, which includes the syntax, semantics, and the concepts that govern it. A single language can have many implementations; the approach may be an interpreter (directly executes the code) or a compiler (converts code into another intermediate language, byte-code or machine-code). CPython is the official implementation and interpreter of Python. Due to this default/official implementation, Python is called an interpreted language, though there are many compiled versions of Python.

Wikipedia defines three execution strategies for interpreters:

> 1. Parse the source code and perform its behavior directly
> 2. Translate source code into some efficient intermediate representation or object code and immediately execute that
> 3. Explicitly execute stored precompiled code made by a compiler which is part of the interpreter system

Python is of the 2nd type. Python is executed in two steps, but the intermediate result (Python byte code, syntax checks made here) is hidden and is immediately executed. Python is dynamically typed; hence **the interpreter has to do some conversion work every time a statement or function is executed**. This conversion makes the language slower than a compiled language, where all the conversions from source code semantics to the machine level are done once and for all.

##### JIT Compilers

A standard compiler would require inferring the data types beforehand. Since Python is dynamically typed, such an inference of data types is probably impossible. This is the reason why dynamically typed languages cannot be purely compiled (please refer to [this](https://stackoverflow.com/questions/18557438/why-cant-pure-python-be-fully-compiled) Stackoverflow question).

A JIT compiler would **compile** the program (Python byte code) as it is running, instead of interpreting every time a method is invoked. This gives it the dynamic runtime information, which was earlier missing in a static compilation. If a JIT compiler has compiled the code: `multiply(2, 3)`, it would not compile `multiply(3, 4)` again. But it will only compile another version if `multiply(1.4, 1.2)` appears. You might think that the additional compilation overhead and interpretation would add to the time taken by the program. But this is all made up by optimizing the compiled code that we generated after the JIT has gathered information from the program (for example, data types that a particular function takes in).

##### XLA

XLA is a particular compiler built specifically for linear algebra-related tasks. It uses the JIT and performs many optimizations to finally produce a complied assembly language output. One of the many benefits is ops fusion. In a typical scenario, we would pull data from memory, perform a single operation, push it back and then pull it again when we need to perform the next operation. XLA compiler would pull all the memory required for once, perform all the operations, and then flush it back to the memory. I found this topic very involved, a lot of this low level details are abstracted away by JAX. Hence I won't go into much depth here. Refer [this](https://www.tensorflow.org/xla) for a detailed overview of XLA


<br>


### References

- [https://github.com/HIPS/autograd](https://github.com/HIPS/autograd)
- [https://carolchen.me/blog/technical/jits-intro/](https://carolchen.me/blog/technical/jits-intro/)
- [https://en.wikipedia.org/wiki/Just-in-time_compilation](https://en.wikipedia.org/wiki/Just-in-time_compilation)
- [https://www.tensorflow.org/xla](https://www.tensorflow.org/xla)
