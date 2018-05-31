package hr.fer.zemris.nnao.geneticAlgorithms.crossovers;

import hr.fer.zemris.nnao.geneticAlgorithms.GAUtil;
import hr.fer.zemris.nnao.geneticAlgorithms.Solution;
import hr.fer.zemris.nnao.neuralNetwork.activations.IActivation;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static hr.fer.zemris.nnao.geneticAlgorithms.GAUtil.integerArrayToIntArrayConverter;
import static hr.fer.zemris.nnao.neuralNetwork.NNUtil.*;

public class SimpleCrossover implements Crossover {

    private static final Random rand = new Random();

    @Override
    public Solution doCrossover(Solution first, Solution second) {

        int splitIndex = Math.min(first.getNumberOfLayers(), second.getNumberOfLayers());
        int crossoverPoint = rand.nextInt(splitIndex);

        List<Integer> layersList = new ArrayList<>();
        List<IActivation> activationsList = new ArrayList<>();

        for (int i = 0; i < crossoverPoint; ++i) {
            layersList.add(first.getLayers()[i]);
            activationsList.add(first.getActivations()[i]);
        }

        for (int i = crossoverPoint; i < second.getNumberOfLayers(); ++i) {
            layersList.add(second.getLayers()[i]);
            activationsList.add(second.getActivations()[i]);
        }

        IActivation[] activationsArray = activationsList.toArray(new IActivation[activationsList.size()]);
        int[] layers = integerArrayToIntArrayConverter(layersList.toArray(new Integer[layersList.size()]));

        return new Solution(activationsArray, layers, layers.length);
    }

}
