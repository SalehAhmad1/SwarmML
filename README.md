# SwarmML
## A python package that utilizes Particle Swarm Optimization for Feature Selection

SwarmML is a Python package that implements Particle Swarm Optimization (PSO) for feature selection. It can be used to improve the performance of classification and regression tasks by automatically selecting the most informative features from a given dataset.

## Features
- Performs feature selection using Particle Swarm Optimization.
- Supports both classification and regression tasks.
- Provides customizable parameters for PSO algorithm.
- Simple and easy to use package.

## Installation
You can install SwarmML using pip:
```
pip install SwarmML
```

## Examples

### Example 1

```
import SwarmML
from sklearn.datasets import load_iris

X,Y = load_iris(return_X_y=True)

obj = SwarmML.FeatureSelection.Particle_Swarm_Optimization('Classification')
Best_Features, Best_Score = obj.run(X, Y)
```