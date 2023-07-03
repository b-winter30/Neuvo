# Neuvo
Neuvo is an automatic neural network architecture search using evolutionary algorithms. Users can decide between a Genetic Algorithm or Grammatical Evolution search with custom novel activation functions.

To create the NeuroBuilder object, users will need to input the 'type' of evolutionary algorithm, eg 'ga' or 'ge' and if they want the evolution to run in eco mode or not.

'''
Neuro = NeuroBuilder(type='ga', eco=False)
'''

# ECO
Eco mode is a novel method of evolving not only the neural networks architecture but also the evolutionary parameters itself, such as: mutation rate, cloning rate, number of generations and the size of the population.

# Inputs

## Selection
Users can currently choose between two selection operators. 
### Tournament Selection 
A method of running several "tournaments" among a few individuals chosen at random from the population, the winners of the tournament go on to reproduce and produce offspring.

### Roulette Selection

Users should run main.py, two standard data loader functions are provided but it is not necessary to use them if you have some other means of data loading. The only requirement is that the data inputted is two numpy arrays, one containing training data and one containing the corresponding data labels.
