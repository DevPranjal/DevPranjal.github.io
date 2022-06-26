---
title: The gradient argument in PyTorch backward
updated: 2022-06-25
---

#### Starting simple

Consider the following definitions of variables and the corresponding gradients with respect to the leaf:

$$a = 5.0$$

$$b = a \times a \implies \frac{db}{da} = 2a$$

$$c = 8 \times b \implies \frac{dc}{da} = \frac{dc}{db} \times \frac{db}{da} = 8 \times 2a = 16a$$

$$d = \log(c) \implies \frac{dd}{da} = \frac{dd}{dc} \times \frac{dc}{db} \times \frac{db}{da} = \frac{1}{c} \times 8 \times 2a = \frac{2}{a}$$

Assume we want to calculate the gradient of $$d$$ with respect to the leaf node $$a$$. This can be done in PyTorch as:

```python
import torch

a = torch.tensor(5., requires_grad=True)
b = a * a
c = 8 * b
d = torch.log(c)

d.backward(retain_graph=True)
a.grad  # 0.4
```

Now, consider $$d$$ as the output but we do not have the definition of $$d$$ (or in some cases, $$d$$ is the black-box). Instead we have the _upstream_ gradients until $$c$$. That is, we have the gradients of $$d$$ with respect to the intermediate $$c$$. We can now use the `gradient` argument of the `.backward()` function to calculate the gradients of $$d$$ (the output) with respect to $$a$$ as follows:

```python
a.grad.zero_()

upstream_grads = torch.tensor(0.005)
c.backward(gradient=upstream_grads, retain_graph=True)
a.grad  # 0.4
```

Similarly, if we extended the _black-box_ to include both $$d$$ and $$c$$ and are given the upstream gradients until $$b$$, we can find the gradients of the output with respect to $$a$$ as follows:

```python
a.grad.zero_()

upstream_grads = torch.tensor(0.04)
b.backward(gradient=upstream_grads, retain_graph=True)
a.grad  # 0.4
```

Thus the `gradient` argument represents the upstream gradients until that point in the neural network. Since the upstream gradients are multiplied (chain rule), they can also be considered the scaling term in the gradient calculation.

#### Extending to higher dimensions

The previous example has now been extended to a higher dimension. Explanations have been ommitted to give you time to understand :)

```python
import torch

a = torch.tensor([1, 2, 3.], requires_grad=True)
b = a * a
c = 8 * b
d = torch.log(c)
e = torch.sum(d)

e.backward(retain_graph=True)
a.grad  # [2.0000, 1.0000, 0.6667]
```

```python
a.grad.zero_()

# d(e) / d(d_i) = 1
upstream_grads = torch.tensor([1, 1, 1])
d.backward(gradient=upstream_grads, retain_graph=True)
a.grad  # [2.0000, 1.0000, 0.6667]
```

```python
a.grad.zero_()

# d(e) / d(c_i) = 1 / c_i
upstream_grads = torch.tensor([1/(8*1*1), 1/(8*2*2), 1/(8*3*3)])
c.backward(gradient=upstream_grads, retain_graph=True)
a.grad  # [2.0000, 1.0000, 0.6667]
```

```python
a.grad.zero_()

# d(e) / d(b_i) = 8 / c_i
upstream_grads = torch.tensor([8/(8*1*1), 8/(8*2*2), 8/(8*3*3)])
b.backward(gradient=upstream_grads, retain_graph=True)
a.grad  # [2.0000, 1.0000, 0.6667]
```

#### A better example

When working with model extraction, we often cascade a black box model in the framework to get the output logits for a particular input. Often, we have a generator model before the teacher to generate meaningful queries to be passed to the teacher. This allows the student (or the clone) to learn from this query-output pair. This framework was presented in [Data-Free Model Extraction](https://arxiv.org/abs/2011.14779) [^1]

<br>

<center>
<figure>
  <img src="../../assets/dfme.png" alt="DFME Framework" style="width:80%">
  <!-- <figcaption>The DFME framework</figcaption> -->
</figure>
</center>

<br>

How do we train the generator? We can backpropagate through the generator graph to get the gradients with respect to the generator output. But how do we backprop through the teacher? Since the teacher is in a black box, we approximate the gradients of the loss (between teacher and student) _with respect to the generator outputs_. And then: use these as the `gradient` argument in the `.backward` call of the generator output to indicate that the approximated gradient is the upstream gradient until the generator output. Take a moment to digest that, and proceed below to understand the code template. Carefully analyze the shape of the approximated gradients:

```python
gen_op = generator(random_noise)    # shape: B * C * H * W
teacher_op = teacher(gen_op)        # shape: B * NC
student_op = student(gen_op)        # shape: B * NC

def approx_grads(gen_op, teacher, student, loss_fn):
    # approximate grads = d(loss(teacher op, student op))
    #                     ------------------------------
    #                           d(generator output)
    return upstream_grads
```

```python
upstream_grads = approx_grads(...)  # shape: B * C * H * W
gen_op.backward(upstream_grads)
# gives gradients of gen parameters wrt loss
```

[^1]: Data-Free Model Extraction: https://arxiv.org/abs/2011.14779