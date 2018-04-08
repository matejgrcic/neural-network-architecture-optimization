package hr.fer.zemris.nnao.geneticAlgorithms;

import hr.fer.zemris.nnao.neuralNetwork.NNUtil;
import hr.fer.zemris.nnao.neuralNetwork.NeuralNetwork;
import hr.fer.zemris.nnao.neuralNetwork.activations.ActivationFunctions;
import hr.fer.zemris.nnao.neuralNetwork.activations.IActivation;
import org.apache.commons.math3.linear.RealMatrix;

import java.util.*;

import static hr.fer.zemris.nnao.neuralNetwork.NNUtil.*;

public class GenerationGA {

    private static final Random rand = new Random();

    private List<Solution> population;

    private int currentIteration = 0;
    private int populationSize;

    public GenerationGA(int populationSize) {
        this.populationSize = populationSize;

    }

    private static Solution[] createInitialPopulation(int populationSize) {
        Solution[] population = new Solution[populationSize];
        for (int i = 0; i < populationSize; ++i) {
            int numberOfLayers = rand.nextInt(6);
            int[] architecture = new int[numberOfLayers];
            for (int j = 0; j < populationSize; ++j) {
                architecture[j] = rand.nextInt(10);
            }
            IActivation[] activations = new IActivation[numberOfLayers];
            activations[0] = ActivationFunctions.Identity;
            activations[numberOfLayers - 1] = ActivationFunctions.Identity;
            IActivation[] allActivations =
                    new IActivation[]{ActivationFunctions.Identity, ActivationFunctions.ReLU, ActivationFunctions.Sigmoid};
            for (int j = 1; j < populationSize - 1; ++j) {
                activations[j] = allActivations[rand.nextInt(allActivations.length)];
            }

            double[] weights = createRandomArray(calculateNumberOfWeights(architecture));
            population[i] = new Solution(activations, numberOfLayers, architecture, weights);
        }

        return population;
    }

    public void run(FitnessCalculator fitnessCalculator, int maxIter) {
        population = new ArrayList<>(Arrays.asList(createInitialPopulation(populationSize)));

        Collections.sort(population, (s1, s2) -> (int) (s1.getFitness() - s2.getFitness()));

        List<Solution> nextGeneration = new ArrayList<>(population.size());
        List<Integer> selectedIndexes = new ArrayList<>(3);
        List<IActivation> activations1 = new ArrayList<>();
        List<IActivation> activations2 = new ArrayList<>();
        List<Integer> architecture1 = new ArrayList<>();
        List<Integer> architecture2 = new ArrayList<>();

        for (int i = 0; i < maxIter; ++i) {
            Collections.sort(population, (s1, s2) -> (int) (s1.getFitness() - s2.getFitness()));
            int attempts = 0;
            while (nextGeneration.size() != populationSize) {
                selectedIndexes.add(rand.nextInt(populationSize));
                selectedIndexes.add(rand.nextInt(populationSize));
                selectedIndexes.add(rand.nextInt(populationSize));
                Collections.sort(selectedIndexes, Comparator.reverseOrder());
                Solution parent1 = population.get(selectedIndexes.get(0));
                Solution parent2 = population.get(selectedIndexes.get(1));
                Solution test = population.get(selectedIndexes.get(2));
                //krizanje
                activations1.clear();
                activations2.clear();
                architecture1.clear();
                architecture2.clear();
                int maxInt = Math.min(parent1.getNumberOfLayers(), parent2.getNumberOfLayers());
                int breakPoint = rand.nextInt(maxInt) + 1;
                activations1.addAll(Arrays.asList(Arrays.copyOfRange(parent1.getActivations(), 0, breakPoint)));
                activations1.addAll(Arrays.asList(Arrays.copyOfRange(parent2.getActivations(), breakPoint, parent2.getActivations().length)));
                activations2.addAll(Arrays.asList(Arrays.copyOfRange(parent2.getActivations(), 0, breakPoint)));
                activations2.addAll(Arrays.asList(Arrays.copyOfRange(parent1.getActivations(), breakPoint, parent1.getActivations().length)));

                int[] a1 = parent1.getArchitecture();
                int[] a2 = parent2.getArchitecture();
                for (int j = 0; j < breakPoint; ++j) {
                    architecture1.add(a1[j]);
                }
                for (int j = breakPoint; j < a2.length; ++j) {
                    architecture1.add(a2[j]);
                }
                for (int j = 0; j < breakPoint; ++j) {
                    architecture2.add(a2[j]);
                }
                for (int j = breakPoint; j < a1.length; ++j) {
                    architecture2.add(a1[j]);
                }


                ++attempts;
            }
        }
    }
}
