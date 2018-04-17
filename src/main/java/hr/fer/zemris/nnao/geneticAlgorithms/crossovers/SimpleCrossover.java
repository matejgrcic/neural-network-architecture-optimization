package hr.fer.zemris.nnao.geneticAlgorithms.crossovers;

import hr.fer.zemris.nnao.geneticAlgorithms.Solution;
import hr.fer.zemris.nnao.neuralNetwork.activations.IActivation;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static hr.fer.zemris.nnao.neuralNetwork.NNUtil.calculateNumberOfWeights;
import static hr.fer.zemris.nnao.neuralNetwork.NNUtil.createRandomArray;

public class SimpleCrossover implements Crossover {

    private static final Random rand = new Random();

    @Override
    public Solution doCrossover(Solution first, Solution second) {

        int splitIndex = Math.min(first.getNumberOfLayers(), second.getNumberOfLayers());
        int crossoverPoint = rand.nextInt(splitIndex);

        List<Integer> layers = new ArrayList<>();
        List<IActivation> activations = new ArrayList<>();

        for (int i = 0; i<crossoverPoint; ++i) {
            layers.add(first.getArchitecture()[i]);
            activations.add(first.getActivations()[i]);
        }

        for (int i = crossoverPoint; i<second.getNumberOfLayers(); ++i) {
            layers.add(second.getArchitecture()[i]);
            activations.add(second.getActivations()[i]);
        }

        IActivation[] activationsArray = new IActivation[activations.size()];
        Integer[] architectureArray = new Integer[layers.size()];

        activationsArray = activations.toArray(activationsArray);
        architectureArray = layers.toArray(architectureArray);

        int[] architecture = new int[architectureArray.length];
        for(int i = 0; i<architectureArray.length; ++i){
            architecture[i] = architectureArray[i];
        }

        double[] weights = createRandomArray(calculateNumberOfWeights(architecture));
        return new Solution(activationsArray,architecture.length,architecture,weights);
    }
}
