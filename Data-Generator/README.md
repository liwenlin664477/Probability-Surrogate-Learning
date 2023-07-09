# Data Generator
This Data Generator is a utility designed to generate datasets for use in training and testing machine learning models. It specializes in generating solutions for a variety of Partial Differential Equations (PDEs), including the heat equation, Poisson's equation, the Burgers' equation, and the Navier-Stokes equation in U-Rectangle form (NavierStockURec).

# Features
Versatile: Capable of generating data for multiple types of PDEs.
Customizable: Allows control over various parameters of the generated dataset.
Efficient: Optimized for speed, enabling large datasets to be generated in a reasonable timeframe.

# Usage
To use the Data Generator, follow these steps:

Install the necessary dependencies.
Run the main script, just specifying the type of PDE.
The generated data will be output to a specified location.
```
data/
├── __raw__/
│   └── Heat_8_128/
│       └──...
├── matlab_solvers
```

# Examples
Here are some examples of how to use the Data Generator:
```
python generate.py -domain=Heat
```

# Contributing
We welcome contributions to the Data Generator project. If you have a feature request, bug report, or proposal for improvement, please open an issue on our GitHub page.
