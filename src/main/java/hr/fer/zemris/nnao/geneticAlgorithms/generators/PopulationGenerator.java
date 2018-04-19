package hr.fer.zemris.nnao.geneticAlgorithms.generators;

import hr.fer.zemris.nnao.geneticAlgorithms.Solution;
import hr.fer.zemris.nnao.geneticAlgorithms.generators.IPopulationGenerator;
import hr.fer.zemris.nnao.neuralNetwork.activations.ActivationFunctions;
import hr.fer.zemris.nnao.neuralNetwork.activations.IActivation;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static hr.fer.zemris.nnao.neuralNetwork.NNUtil.calculateNumberOfWeights;
import static hr.fer.zemris.nnao.neuralNetwork.NNUtil.createRandomArray;

public class PopulationGenerator implements IPopulationGenerator {

    private static final Random rand = new Random();

    private int minLayersNum;
    private int maxLayersNum;
    private int minLayerSize;
    private int maxLayerSize;
    private int inputLayerSize;
    private int outputLayerSize;

    public PopulationGenerator(int minLayersNum, int maxLayersNum, int minLayerSize, int maxLayerSize, int inputLayerSize, int outputLayerSize) {
        this.minLayersNum = minLayersNum;
        this.maxLayersNum = maxLayersNum;
        this.minLayerSize = minLayerSize;
        this.maxLayerSize = maxLayerSize;
        this.inputLayerSize = inputLayerSize;
        this.outputLayerSize = outputLayerSize;
    }

    public List<Solution> createInitialPopulation(int populationSize) {
        List<Solution> population = new ArrayList<>(populationSize);
        for (int i = 0; i < populationSize; ++i) {
            int numberOfLayers = rand.nextInt(maxLayersNum - minLayersNum + 1) + minLayersNum;
            int[] architecture = new int[numberOfLayers];
            for (int j = 0; j < numberOfLayers; ++j) {
                architecture[j] = rand.nextInt(maxLayerSize - minLayerSize + 1) + minLayerSize;
            }
            IActivation[] activations = new IActivation[numberOfLayers];
            activations[0] = ActivationFunctions.Identity;
            activations[numberOfLayers - 1] = ActivationFunctions.Identity;
            IActivation[] allActivations = ActivationFunctions.allActivations;

            for (int j = 1; j < numberOfLayers; ++j) {
                activations[j] = allActivations[rand.nextInt(allActivations.length)];
            }

            architecture[0] = inputLayerSize;
            architecture[architecture.length - 1] = outputLayerSize;

            double[] weights = createRandomArray(calculateNumberOfWeights(architecture));
            population.add(new Solution(activations, numberOfLayers, architecture, weights));
        }

        return population;
    }

    @Override
    public Solution createIndividual() {
        int numberOfLayers = rand.nextInt(maxLayersNum - minLayersNum + 1) + minLayersNum;
        int[] architecture = new int[numberOfLayers];
        for (int j = 0; j < numberOfLayers; ++j) {
            architecture[j] = rand.nextInt(maxLayerSize - minLayerSize + 1) + minLayerSize;
        }
        IActivation[] activations = new IActivation[numberOfLayers];
        activations[0] = ActivationFunctions.Identity;
        activations[numberOfLayers - 1] = ActivationFunctions.Identity;
        IActivation[] allActivations = ActivationFunctions.allActivations;

        for (int j = 1; j < numberOfLayers; ++j) {
            activations[j] = allActivations[rand.nextInt(allActivations.length)];
        }

        architecture[0] = inputLayerSize;
        architecture[architecture.length - 1] = outputLayerSize;

        double[] weights = createRandomArray(calculateNumberOfWeights(architecture));
        return new Solution(activations, numberOfLayers, architecture, weights);
    }
}
