# KMeans from Scratch

## Overview
This project implements several K-means algorithms from scratch in Python. Designed to offer a deeper understanding of the inner workings of these popular clustering techniques, the implementation focuses on clarity and educational value.

## Features
- **KMeans Implementations**: Three variations of the K-means algorithm are provided, including the standard K-means with random initialization, the KMeans++ with an improved initialization method, and the Soft K-means with a probabilistic, or fuzzy, approach to cluster assignment.
- **Customizable Parameters**: Users can specify the number of clusters, the initialization method, and other parameters.
- **Data Visualization**: The project includes functionality to visualize the clustering results, making it easier to understand and analyze the output.

## Installation
To use this project, clone the repository from GitHub and ensure you have Python installed on your machine.

```bash
git clone https://github.com/Ger2720/Kmeans.git
cd Kmeans
```

## Usage
Import the KMeans class from the script and initialize it with your dataset and desired parameters.

```python
from Kmeans_from_scratch import KMeans

# Example usage
kmeans = KMeans(data=my_dataset, k=3, init='kmeans++')
kmeans.fit()
```
