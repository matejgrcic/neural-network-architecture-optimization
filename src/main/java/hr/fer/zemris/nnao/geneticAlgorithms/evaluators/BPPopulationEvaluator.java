package hr.fer.zemris.nnao.geneticAlgorithms.evaluators;

import hr.fer.zemris.nnao.bp.Backpropagation;
import hr.fer.zemris.nnao.datasets.DatasetEntry;
import hr.fer.zemris.nnao.geneticAlgorithms.Solution;
import hr.fer.zemris.nnao.neuralNetwork.NNUtil;
import hr.fer.zemris.nnao.neuralNetwork.NeuralNetwork;
import org.apache.commons.math3.linear.RealMatrix;

import java.util.List;

public class BPPopulationEvaluator extends AbstractPopulationEvaluator {

    private List<DatasetEntry> trainingDataset;
    private List<DatasetEntry> validationDataset;
    private double learningRate;
    private long maxIteration;
    private double desiredError;
    private double desiredPrecision;
    private int batchSize;
    private int maxTrys;

    private double errorFactor;
    private double weightsFactor;
    private double layersFactor;

    public BPPopulationEvaluator(List<DatasetEntry> dataset, double learningRate, long maxIteration, double desiredError,
                                 double desiredPrecision, int batchSize, double trainingSetPercentage, int maxTrys,
                                 double errorFactor, double weightsFactor, double layersFactor) {

        int splitIndex = (int) Math.round(trainingSetPercentage * dataset.size());
        trainingDataset = dataset.subList(0, splitIndex);
        validationDataset = dataset.subList(splitIndex, dataset.size());
        this.learningRate = learningRate;
        this.batchSize = batchSize;
        this.maxIteration = maxIteration;
        this.desiredError = desiredError;
        this.desiredPrecision = desiredPrecision;
        this.maxTrys = maxTrys;

        this.layersFactor = layersFactor;
        this.weightsFactor = weightsFactor;
        this.errorFactor = errorFactor;
    }

    @Override
    public double evaluateSolution(Solution solution) {
        double bestFitness = Double.MAX_VALUE;
        RealMatrix[] bestWeights = null;
        NeuralNetwork nn = new NeuralNetwork(solution.getLayers(), solution.getActivations());
        for(int i = 0; i<maxTrys; ++i) {
            nn.setWeights(solution.getWeights());
            Backpropagation bp = new Backpropagation(trainingDataset, validationDataset,
                    learningRate, maxIteration, desiredError, desiredPrecision, nn, batchSize);
            double fitness =  bp.run();
            if(fitness < bestFitness) {
                bestFitness = fitness;
                bestWeights = nn.getWeightsMatrix();
            }
        }

        double totalFitness = layersFactor * nn.getNeuralNetworkArchitecture().length +
                weightsFactor * nn.getWeightsNumber() + errorFactor * bestFitness;

        solution.setFitness(totalFitness);
        solution.setWeights(NNUtil.getWeights(nn.getWeightsNumber(), bestWeights));
        notifyObservers(totalFitness);
        return totalFitness;
    }

}
