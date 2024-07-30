## Required Functions:
* fuel optimization function
* function that calculates cost breakdown
* function to export submission
* hook up hyper parameters to UI

## UI Components

* detailed cost breakdown:
    * need to total cost per size
    * total cost per cost term
    * cost per year

* vehichle type breakdown
    * per year
    * per type

* fuel type breakdown
    * per year
    * per type

* constraints checker

## Requirements for level 2 submission
Submissions for Level 2 must be in a digital format and include:

1. A comprehensive document in PDF format which outlines the model’s methodology, assumptions, data sources and limitations;
2. A slide deck which explains the model’s functionality, performance, business model and business impact;
3. A web link to the prototype of the Concept;
4. A video screen capture of the executable files in the .mp4 format;
5. The deadline for Level 2 Submissions will be by the end of 4 August 2024 Indian Standard Time and late Submissions will not be accepted.

## Judging Criteria
Contestants will be shortlisted based on the criteria below:
1. Functionality and Usability:
    * Does the prototype of the Concept effectively demonstrate the mathematical model for optimizing fleet decarbonization strategies?
    * Is the prototype of the Concept fully functional and does it perform as expected?
    * Is the prototype of the Concept user-friendly and easy to navigate?
    * Are there clear instructions or guidance on how users should to interact with the prototype of the Concept?
2. Innovation:
    * Innovation of modelling techniques.
    * Does the prototype of the Concept showcase innovative approaches or techniques in fleet decarbonization optimization?
    * Are there unique features or functionalities that set the prototype of the Concept apart from existing solutions?
3. Scalability and Performance:
    * Can the prototype of the Concept be scaled to accommodate larger fleets or different scenarios?
    * Does the prototype of the Concept demonstrate scalability in terms of computational resources and data handling?
    * How efficiently does the prototype of the Concept execute the optimization algorithms?
    * Are there any performance bottlenecks or limitations encountered during testing?

# 1. Introduction
## 1.1 Background

Professional, delivery, and operational fleets play a significant role in the global supply chain, offering flexibility, door-to-door service, and connectivity between cities and towns. However, these fleets are also a major contributor to global greenhouse gas emissions. Fleet owners face the challenge of transitioning to net-zero emissions while maintaining business sustainability and customer satisfaction. This transition involves a complex decision-making process that must account for various factors such as timing, location, and approach.

In this competition, the primary challenge is to develop an optimal fleet decarbonization strategy. This involves solving a non-linear optimization problem characterized by a large number of decision variables. The complexity of the problem arises from the need to balance multiple objectives, such as minimizing carbon emissions, meeting customer demand, and controlling operational costs.

Non-linear optimization problems with a large number of decision variables are inherently difficult to solve. The non-linearity introduces complexities that prevent straightforward solutions, requiring specialized algorithms. Moreover, the high dimensionality of the decision space means that there are numerous possible configurations and combinations of decision variables to consider, making the search for an optimal solution computationally intensive.

The optimization model must account for various constraints and objectives simultaneously. These include emission constraints, vehicle capabilities, fuel consumption rates, and other operational costs. Addressing this challenge requires robust optimization techniques capable of handling the non-linearity and the vast decision space. 

Several advanced optimization techniques can be considered for this purpose. Potential algorithms include particle swarm optimization, simulated annealing, and genetic algorithms. Each of these techniques offers unique strengths in exploring and exploiting the search space to find optimal or near-optimal solutions.

This document discusses the choice of using a genetic algorithm to address the optimization problem. The methodology, assumptions, data sources, and limitations of this approach are detailed, demonstrating how genetic algorithms can effectively balance the competing objectives and constraints in fleet decarbonization strategies.

## 1.2 Objectives

* The solution should converge to an exceptable cost value within a reasonable amount of time
* The solution should be scalable to accommodate larger fleets and longer time periods.
* The solution should be scalable to accommodate new constraints.
* The solution should be packaged in a user friendly way.


# 2. Methodology

## 2.1  Why a Genetic Algorithm?

Genetic algorithms are per

Genetic algorithms (GAs) are particularly effective for non-linear optimization in large decision spaces due to several inherent characteristics and strengths:

Global Search Capability:

GAs perform a global search rather than a local search. This means they explore a wide range of potential solutions in the search space, which helps in avoiding local optima—a common issue in non-linear optimization problems where the solution landscape can have multiple peaks and valleys.
Population-Based Approach:

Instead of focusing on a single solution, GAs work with a population of solutions. This diversity in the population helps in exploring different regions of the search space simultaneously. It enhances the ability to discover global optima in complex landscapes with many local optima.
Robustness to Irregular Search Spaces:

Non-linear optimization problems often have complex, irregular, and discontinuous search spaces. GAs do not require gradient information or smoothness in the search space, making them suitable for such problems where traditional optimization methods may fail.
Stochastic Nature:

GAs incorporate randomness through operations like mutation and crossover, which helps in exploring new and uncharted areas of the search space. This stochastic nature prevents premature convergence and ensures a thorough exploration of potential solutions.
Adaptability:

The adaptive nature of GAs allows them to adjust the search process dynamically based on the fitness of the population. This adaptability is crucial in large decision spaces where the search landscape can change dramatically from one region to another.
Parallelism:

GAs are inherently parallel since they evaluate multiple solutions simultaneously. This parallelism can be exploited to speed up the search process, especially in large decision spaces where evaluating all possible solutions sequentially would be computationally prohibitive.


## 2.2 Overview of Genetic Algorithms

A genetic algorithm (GA) is an optimization technique inspired by the principles of natural selection and genetics. It is particularly well-suited for solving complex optimization problems with large and non-linear search spaces. The key principles and operations of a genetic algorithm are selection, crossover, and mutation.

Key Principles and Operations
### Selection:
The selection process is analogous to natural selection where the fittest individuals are chosen to reproduce. In a genetic algorithm, a fitness function evaluates each individual in the population, assigning a fitness score based on how well it solves the problem. The selection operation then chooses individuals for reproduction, typically giving higher probability to those with better fitness scores. Common selection methods include roulette wheel selection, tournament selection, and rank-based selection.

### Crossover:
Crossover is a genetic operator used to combine the genetic information of two parent individuals to produce new offspring. This operation mimics biological reproduction. The crossover process typically involves selecting a crossover point on the parents' chromosome and exchanging the segments beyond this point between the two parents. Common crossover techniques include one-point crossover, two-point crossover, and uniform crossover. The goal is to produce offspring that inherit the best traits from both parents.

### Mutation:
Mutation introduces genetic diversity into the population by randomly altering the genes of individuals. This operation prevents the algorithm from becoming stuck in local optima by ensuring a wider exploration of the search space. Mutation is typically performed with a low probability, modifying one or more genes in an individual's chromosome. Common mutation techniques include bit-flip mutation for binary representations and Gaussian mutation for real-valued representations.
Real-World Applications and References
Genetic algorithms are used in various fields, including engineering, economics, bioinformatics, and artificial intelligence, for tasks such as scheduling, design optimization, and machine learning.



## 2.3 Algorithm Design

The 

Describe the specific design of your genetic algorithm, including:
### 2.2.1 Encoding: 
How solutions are represented (e.g., binary strings, real numbers).
### 2.2.2 Population Initialization: 
How the initial population is generated.
### 2.2.3 Selection Method: 
The technique used to select parents (e.g., roulette wheel, tournament selection).
### 2.2.4 Crossover Method: 
How crossover is performed (e.g., one-point, two-point, uniform crossover).
### 2.2.5 Mutation Method: 
How mutations are introduced into the offspring.
### 2.2.6 Fitness Function: 
How the fitness of each individual is evaluated.

## 2.4 Algorithm Implementation

Provide details about the implementation of the algorithm, including:
Programming Language and Tools: The software and libraries used.
Algorithm Flowchart: A flowchart or pseudocode to illustrate the algorithm's process.
Parameters and Hyperparameters: The key parameters and their chosen values (e.g., population size, crossover rate, mutation rate).

# 3. Assumptions
List and explain any assumptions made in the development and implementation of the genetic algorithm. This could include assumptions about the problem domain, data, and model constraints.
# 4. Data Sources
## 4.1 Data Collection

Describe the data required for your genetic algorithm, including:
Source of Data: Where the data comes from (e.g., datasets, simulations).
Data Characteristics: Key characteristics and attributes of the data.
## 4.2 Data Preprocessing

Explain any preprocessing steps taken to prepare the data for the genetic algorithm. This could include normalization, cleaning, and transformation.
# 5. Results
## 5.1 Experiments and Evaluation

Describe the experiments conducted to evaluate the genetic algorithm, including:
Experimental Setup: The setup of the experiments, including the environment and configurations.
Performance Metrics: The metrics used to evaluate the algorithm's performance (e.g., accuracy, convergence rate).
## 5.2 Results Analysis

Present and analyze the results of your experiments. Include:
Graphs and Tables: Visualizations to illustrate the performance of the algorithm.
Comparison with Baseline: Compare the genetic algorithm's results with a baseline or other methods if applicable.
# 6. Limitations
Discuss the limitations of your genetic algorithm, including:
Model Limitations: Limitations related to the design and assumptions of the model.
Data Limitations: Any limitations related to the data used.
Computational Limitations: Constraints related to computational resources and efficiency.
# 7. Conclusion
Summarize the key findings and contributions of your work. Discuss the implications of your results and suggest possible future work or improvements to the genetic algorithm.
# 8. References
Goldberg, D. E. (1989). Genetic Algorithms in Search, Optimization, and Machine Learning. Addison-Wesley.

Mitchell, M. (1998). An Introduction to Genetic Algorithms. MIT Press.

Holland, J. H. (1975). Adaptation in Natural and Artificial Systems. University of Michigan Press.