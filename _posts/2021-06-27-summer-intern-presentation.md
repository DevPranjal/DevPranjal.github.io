---
title: Summer Intern '22
updated: 2022-06-27
visible: 1
---

#### Initial Directions

- Problems faced at the Inter IIT Tech Meet
    - Mode collapse
    - Poor attack quality
- Exploring Jacobian Based Data Augmentations
    - To improve the quality of attack vectors
    - Augmenting seed data based on Jacobian
- _Dumped in favor of exploring dynamic neural networks_

#### Dynamic Neural Networks

- Can adapt their structures and parameters to different inputs
- Advantages in:
    - Accuracy
    - Computational efficiency
    - Adaptiveness

<center>
<figure>
  <img src="../../assets/dynamic-moe.png" alt="Dynamic Mixture of Experts" style="width:80%">
  <!-- <figcaption>The DFME framework</figcaption> -->
</figure>
</center>

#### Dynamic NNs in Model Extraction

- Can adapt to different target / black-box models
- Can give the best clone model:
    - Smaller target gives rise to smaller clone
- Optimally: guess the target model architecture from the dynamic connections that arise

#### Background and Survey

- Data-Free Model Extraction (https://arxiv.org/pdf/2011.14779.pdf)
    - GANs for synthetic data
    - Gradient estimation for teacher backprop
- Dynamic Substitute Training for Data-Free Black Box Attacks (https://arxiv.org/pdf/2204.00972.pdf)
    - Dynamic clone
    - Special loss function

#### Code

- Dynamic Substitute Training for Data-Free Black Box Attacks
    - Open-source code was not available
- Implemented the code for the above work
    - Plugged it into DFME codebase
- Organized as:
  `/checkpoints` | saved checkpoints
  `/dfme` | dfme code simplified 
  `/models`	| standard architectures and dynamic networks
  `/experiments` | scripts for the experiments conducted

#### Experiments

- Explored various dynamic architectures:
    - Layer skipping
        - Gated ResNets
        - Skip Nets
    - Cascading Networks and Mixture of Experts
    - Dynamic Convolutions
- Inferring target model architecture from dynamic gates of an extracted gated ResNet
    - FAILED
    - But still optimistic

#### Results

Number of queries | 10M (20M used in the DFME paper)
Dataset | CIFAR10

- **Gated ResNets**

  | Teacher | Static Student | Dynamic Student |
  | ------- | -------------- | --------------- |
  | ResNet 18 | 46.24 | 48.41 |
  | ResNet 34 | **41.09** | **47.24** |
  | ResNet 50 | 40.12 | 42.75 |

- **Comparing Gate Connections**

  | Teacher | Test Accuracy | Gate Activations |
  | ------- | ------------- | ---------------- |
  | ResNet 18 | 48.41 | `x-xxx-xx` |
  | ResNet 34 | 47.24 | `xxxx--x-` |
  | ResNet 50 | 42.75 | `----xxx-` |
  | MobileNet V2 | 51.24 |  |
  | Inception V3 | 42.53 | `--xx-x--` |
  | GoogleNet | 42.14 | `--xxx---` |

- **Other Dynamic Techniques**

  | Student Architecture | Test Accuracy |
  | -------------------- | ------------- |
  | Baseline Resnet 18 8x | 41.09 |
  | Gated Resnet 18 8x | 47.24 |
  | Resnet 18 8x cascaded with Resnet 50 (based on logit threshold) | 41.14 |
  | Dynamic convolutions in Resnet 18 8x | 45.13 |
  | Skip Nets | Could not explore |

#### Next?

- Explore and quantify relations between dynamism and the teacher / target
    - First step: exploring gates
    - Next: attention weights in the dynamic conv
- Skip Net analysis:
    - Paper gives analysis for tough vs easy inputs
    - We can step further with a goal for tough vs easy models
- Use cases in edge devices

