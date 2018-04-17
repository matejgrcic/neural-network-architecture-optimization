package hr.fer.zemris.nnao.geneticAlgorithms;

import hr.fer.zemris.nnao.bp.Backpropagation;
import hr.fer.zemris.nnao.datasets.DatasetEntry;
import hr.fer.zemris.nnao.neuralNetwork.NNUtil;
import hr.fer.zemris.nnao.neuralNetwork.NeuralNetwork;

import java.util.ArrayList;
import java.util.List;

public class BPPopulationEvaluator implements PopulationEvaluator {

    private List<DatasetEntry> trainingDataset;
    private List<DatasetEntry> validationDataset;
    private double learningRate;
    private long maxIteration;
    private double desiredError;
    private double desiredPrecision;
    private int batchSize;
    private double trainingSetPercentage;


    public BPPopulationEvaluator(List<DatasetEntry> dataset, double learningRate, long maxIteration, double desiredError,
                                 double desiredPrecision, int batchSize, double trainingSetPercentage) {

        int splitIndex = (int)Math.round(trainingSetPercentage * dataset.size());
        trainingDataset = dataset.subList(0, splitIndex);
        validationDataset = dataset.subList(splitIndex, dataset.size());
        this.learningRate = learningRate;
        this.batchSize = batchSize;
        this.maxIteration = maxIteration;
        this.desiredError = desiredError;
        this.desiredPrecision = desiredPrecision;

    }

    public double evaluateSolution(Solution solution) {
        NeuralNetwork nn = new NeuralNetwork(solution.getArchitecture(),solution.getActivations());
        nn.setWeights(solution.getWeights());
        Backpropagation bp = new Backpropagation(trainingDataset, validationDataset,
                learningRate, maxIteration, desiredError, desiredPrecision, nn, batchSize);

        return bp.run();
    }

}
