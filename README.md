# Neuvo
Neuvo is an automatic neural network architecture search using evolutionary algorithms. Users can decide between a Genetic Algorithm or Grammatical Evolution search with custom novel activation functions.

Users should run example.py, two standard example data loader functions are provided for 1D and 2D data but it is not necessary to use them if you have some other means of data loading. The only requirement is that the data inputted is two numpy arrays, one containing training data and one containing the corresponding data labels.

To create the NeuroBuilder object, users will need to input the 'type' of evolutionary algorithm, eg 'ga' or 'ge' and if they want the evolution to run in eco mode or not.

```Neuro = NeuroBuilder(type='ga', eco=False)```

# ECO
Eco mode is a novel method of evolving not only the neural networks architecture but also the evolutionary parameters itself, such as: mutation rate, cloning rate, number of generations and the size of the population.

# Inputs

## Selection
Users can currently choose between two selection operators. 

### Tournament Selection 
A method of running several "tournaments" among a few individuals chosen at random from the population, the winners of the tournament go on to reproduce and produce offspring.

```Neuro.selection = 'Tournament'```

### Roulette Selection
A roulette wheel is constructed from the relative fitness (ratio of individual fitness and total fitness) of each individual, all individuals here have a chance of reproducing, but individuals with a higher relative fitness have a greater chance of being selected.

```Neuro.selection = 'Roulette'```

## Evolutionary parameters
If the user is not using eco mode, they will need to set these manually as shown below.
### Population size
This parameter sets the maximum size for the population.

```Neuro.population_size = 10```
### Mutation rate
This represents the percentage chance that an individual will mutate.

```Neuro.mutation_rate = 0.01```
### Cloning rate
This represents the percentage chance that an individual will be cloned into the next generation without reproduction.

```Neuro.cloning_rate = 0.3```

### Max generations
The maximum number of generations the evolutionary process will run for.

```Neuro.max_generations = 500```

## Verbose mode
Neuvo's backend runs with Tensorflow, therefore for users to track their training, they can set the Verbose flag via this variable.

```Neuro.verbose = 2```

## Output variables
Neuvo creates output files to record the results of the evolutionary process, to do this, it requires a name you would like to save the file as.

```Neuro.dataset_name = 'Cifar'```

# Load data
Neuvo can create CNN's and ANN's for classifying 2D or 1D data, simply pass a list containing two numpy arrays, one containg the training data (X) and one containing the corresponding data labels (Y).

```Neuvo.load_data([X, Y])```

# Setting a fitness function
There are multiple fitness functions one can choose from. They include:

* f1
* precision
* recall
* mae
* rmse
* accuracy
* validation_accuracy
* speed
* val_acc_x_f1

```Neuro.set_fitness_function('f1')```

# Initialise the population
This function initialises the population by creating NeuroEvolution objects and running them for classification depending on the shape of the inputted data (ANN/CNN).

This function also allows for users to insert a phenotype if running with GA set, or input a genotype if running with GE set.

These insertions allow the user to include a 'prior' best network should they have one.

## Inserting with GA
When using Neuvo with GA set, we have to insert with a phenotype, a phenotype in Neuvo is represented as a dictionary. Therefore we insert a list of dictionaries, one dictionary for each phenotype we would like inserted into the population.

```insertions = [{ 'hidden layers' : 2, 'nodes' : 8, 'activation functions' : ['relu', 'relu',  'relu', 'sigmoid'], 'optimiser' : 'Adam', 'number of epochs' : 5, 'batch size' : 2}]```

```Neuro.initialise_pop(insertions=insertions)```

## Inserting with GE
Due to the inherent mapping function that Grammatical Eviolution utilises with its genotype-to-phenotype mapping, to insert an individual into the population we have to insert it as a genotype.

```insertions = [[25, 36, 27, 38, 9, 33, 30, 29, 11, 2, 35, 12, 39, 22, 16, 6, 19, 21, 3, 4, 8, 17, 37, 28, 1, 15, 31, 10, 14, 0, 24, 20]]```

```Neuro.initialise_pop(insertions=insertions)```

When inserting with GE it is important to note that the phenotypes created are reliant on the quality of the grammar.


