package hr.fer.zemris.nnao.geneticAlgorithms;

import hr.fer.zemris.nnao.datasets.DatasetEntry;
import hr.fer.zemris.nnao.neuralNetwork.NNUtil;
import hr.fer.zemris.nnao.neuralNetwork.NeuralNetwork;

import java.util.List;

public class FitnessCalculator {

    private static NeuralNetwork nn = new NeuralNetwork();

    public FitnessCalculator() {
    }

    public double evaluateSolution(Solution solution) {
        nn.setWeights(NNUtil.createWeightMatrices(solution.getArchitecture(),solution.getWeights()));
        nn.setActivationFunctions(solution.getActivations());

        return 0.;
    }

}
