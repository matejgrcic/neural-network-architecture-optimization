package hr.fer.zemris.nnao.geneticAlgorithms;

import hr.fer.zemris.nnao.datasets.DatasetEntry;
import hr.fer.zemris.nnao.datasets.DatasetUtils;
import hr.fer.zemris.nnao.neuralNetwork.NeuralNetwork;
import hr.fer.zemris.nnao.swarmAlgorithms.AlgorithmPSO;

import java.util.List;
import java.util.function.BiFunction;

public class PSOPopulationEvaluator implements PopulationEvaluator {

    private List<DatasetEntry> dataset;
    private int populationSize;
    private int maxIterations;
    private double desiredError;
    private double desiredPrecision;

    public PSOPopulationEvaluator(List<DatasetEntry> dataset, int populationSize, int maxIterations, double desiredError, double desiredPrecision) {
        this.dataset = dataset;
        this.populationSize = populationSize;
        this.maxIterations = maxIterations;
        this.desiredError = desiredError;
        this.desiredPrecision = desiredPrecision;
    }

    @Override
    public double evaluateSolution(Solution solution) {
        NeuralNetwork nn = new NeuralNetwork(solution.getArchitecture(), solution.getActivations());
        nn.setWeights(solution.getWeights());
        double[] lowerBound = new double[nn.getWeightsNumber()];
        for (int i = 0; i < lowerBound.length; ++i) {
            lowerBound[i] = -10;
        }
        double[] upperBound = new double[nn.getWeightsNumber()];
        for (int i = 0; i < lowerBound.length; ++i) {
            lowerBound[i] = 10.;
        }
        double[] lowerSpeed = new double[nn.getWeightsNumber()];
        for (int i = 0; i < lowerBound.length; ++i) {
            lowerBound[i] = -1.;
        }
        double[] upperSpeed = new double[nn.getWeightsNumber()];
        for (int i = 0; i < lowerBound.length; ++i) {
            lowerBound[i] = 1.;
        }

        BiFunction<Double, Double, Boolean> comparator = (t, u) -> Math.abs(t) > Math.abs(u);


        AlgorithmPSO pso = new AlgorithmPSO(populationSize, nn.getWeightsNumber(), lowerBound, upperBound, lowerSpeed, upperSpeed);
        double[] result = pso.run(t -> {
            nn.setWeights(t);
            double sum = 0.;
            for (DatasetEntry d : dataset) {
                sum += Math.abs(nn.forward(d.getInput())[0] - d.getOutput()[0]);
            }
            return sum/dataset.size();
        }, comparator, desiredError, desiredPrecision, maxIterations);

        nn.setWeights(result);
        solution.setWeights(result);
        double sum = 0.;
        for (DatasetEntry d : dataset) {
            sum += Math.abs(nn.forward(d.getInput())[0] - d.getOutput()[0]);
        }
        return sum / dataset.size();
    }
}
