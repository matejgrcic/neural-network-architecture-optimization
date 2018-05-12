package hr.fer.zemris.nnao.geneticAlgorithms.mutations;

import hr.fer.zemris.nnao.geneticAlgorithms.Solution;
import hr.fer.zemris.nnao.neuralNetwork.activations.ActivationFunctions;
import hr.fer.zemris.nnao.neuralNetwork.activations.IActivation;

import java.util.Random;

import static hr.fer.zemris.nnao.neuralNetwork.NNUtil.*;

public class SimpleMutation implements Mutation {

    private static final Random rand = new Random();

    private int minLayerSize;
    private int maxLayerSize;
    private int minLayerNumber;
    private int maxLayerNumber;

    private double layerSizeMutationProbability;
    private double layerAdditionMutationProbability;
    private double layerRemoveMutationProbability;
    private double activationMutationProbability;

    public SimpleMutation(int minLayerSize, int maxLayerSize, int minLayerNumber,
                          int maxLayerNumber, double layerSizeMutationProbability,
                          double layerAdditionMutationProbability, double layerRemoveMutationProbability,
                          double activationMutationProbability) {
        this.minLayerSize = minLayerSize;
        this.maxLayerSize = maxLayerSize;
        this.minLayerNumber = minLayerNumber;
        this.maxLayerNumber = maxLayerNumber;
        this.layerSizeMutationProbability = layerSizeMutationProbability;
        this.layerAdditionMutationProbability = layerAdditionMutationProbability;
        this.layerRemoveMutationProbability = layerRemoveMutationProbability;
        this.activationMutationProbability = activationMutationProbability;
    }

    @Override
    public Solution mutate(Solution solution) {

        // mutate layers
        if (rand.nextDouble() < layerSizeMutationProbability) {
            int index = rand.nextInt(solution.getLayers().length - 2) + 1;
            solution.getLayers()[index] = createRandomLayer(minLayerSize, maxLayerSize);
        }
        // mutate activations
        if (rand.nextDouble() < activationMutationProbability) {
            int index = rand.nextInt(solution.getActivations().length);
            solution.getActivations()[index] = createRandomActivation();
        }

        // add layer
        if (solution.getNumberOfLayers() < maxLayerNumber && rand.nextDouble() < layerAdditionMutationProbability) {
            int index = rand.nextInt(solution.getLayers().length - 2) + 1;
            int[] layers = new int[solution.getNumberOfLayers() + 1];
            System.arraycopy(solution.getLayers(), 0, layers, 0, index);
            layers[index] = createRandomLayer(minLayerSize, maxLayerSize);
            System.arraycopy(solution.getLayers(), index, layers,
                    index + 1, solution.getNumberOfLayers() - index);

            IActivation[] activationsArr = new IActivation[solution.getActivations().length + 1];
            System.arraycopy(solution.getActivations(), 0, activationsArr, 0, index);
            activationsArr[index] = createRandomActivation();
            System.arraycopy(solution.getActivations(), index, activationsArr,
                    index + 1, solution.getNumberOfLayers() - index);

            solution.setNumberOfLayers(layers.length);
            solution.setArchitecture(layers, activationsArr);
        }

        // remove layer
        if (solution.getNumberOfLayers() > minLayerNumber && rand.nextDouble() < layerRemoveMutationProbability) {
            int index = rand.nextInt(solution.getLayers().length - 2) + 1;
            int[] layers = new int[solution.getNumberOfLayers() - 1];
            System.arraycopy(solution.getLayers(), 0, layers, 0, index);
            System.arraycopy(solution.getLayers(), index + 1, layers,
                    index, solution.getNumberOfLayers() - index - 1);

            IActivation[] activationsArr = new IActivation[solution.getActivations().length - 1];
            System.arraycopy(solution.getActivations(), 0, activationsArr, 0, index);
            System.arraycopy(solution.getActivations(), index + 1, activationsArr,
                    index, solution.getNumberOfLayers() - index - 1);

            solution.setNumberOfLayers(layers.length);
            solution.setArchitecture(layers, activationsArr);
        }

        double[] weights = getWeights(
                calculateNumberOfWeights(solution.getLayers()), createWeightMatrices(solution.getLayers()));
        solution.setWeights(weights);
        return solution;
    }

    private static int createRandomLayer(int minLayerSize, int maxLayerSize) {
        return rand.nextInt(maxLayerSize - minLayerSize + 1) + minLayerSize;
    }

    private static IActivation createRandomActivation() {
        return ActivationFunctions.allActivations[rand.nextInt(ActivationFunctions.allActivations.length)];
    }
}
