# Exploring neural representations of spatial environments and navigation with artificial agents
Sandbox environment simulating cognitive maps with graphical neural networks and deep reinforcement learning to explore human-ai spatial learning and ecosystem collaboration.

# Intro
### Constructing cell activity patterns (grid cells)
We use sinusoidal interference to simulate hexagonal patterns in the brain (this is a 
simplified, supervised learning model of the brain's neural firing processes which are
naturally nondeterministic and unsupervised):

```math
f(x,y) = \sum_{i=1}^{3} \cos(k_i \cdot (x,y) + \phi_i)
```

where:
- $`(x,y)`$ are the coordinates of the agent.
- $`k_i`$ represents the frequency of neural firing.
- $`\phi_i`$ is the phase shift/offset between waves, tracking shifts to create an optimal hexagonal representation that reduces redundancy.

three waves (representing equilateral triangles that form the hexagon shape) are spaced 120° apart to form a consistent hexagonal pattern.
### Selecting landmark locations and activation patterns (place cells)
Place cells activate at high-density regions called "place fields". The number of unique
place cells locations represents the resolution of environment representation.
<br/>
Activity follows a gaussian distribution (smooth activation profile) with a peak at the center of the place field and decays toward the edges:
```math
A(x) = e^{-\frac{\| x - x_c \|^2}{2\sigma^2}}
```

where:
 - $`A(x)`$ is the activation at position $`x`$.
  - $`x_c`$ is the place cell's preferred location (center of its receptive field).
  - $`\sigma`$ controls the **width** of the place field (how quickly firing decays).


# Getting started

# Dataset
Link: [Population Dynamics Embedding](https://www.kaggle.com/datasets/veeralakrishna/population-dynamics-embeddings). Vector embeddings of locations based on human interactions.
- Polygon coordinates for boundaries
- Topological map of relationships between spaces
- Semantic labels for locations
# References
- [My notes on spatial navigation + graphical neural nets](https://www.remnote.com/a/notes-on-neural-maps-and-spatial-navigation/67c22f7a3df60be1a78d6436)
- [Creating spatial maps](https://www.youtube.com/watch?v=iV-EMA5g288)
### Future work
- Use lateral inhibition as opposed to inhibition to more accurately reflect impact of neural firing on neighboring neurons
- Use a Recurrent neural network to update parameters based on historical data instead of only
on current data
- Train an unsupervised model to learn grid cell patterns with introducing primed input