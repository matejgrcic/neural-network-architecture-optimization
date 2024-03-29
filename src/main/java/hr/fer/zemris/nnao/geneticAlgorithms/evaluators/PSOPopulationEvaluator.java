package hr.fer.zemris.nnao.geneticAlgorithms.evaluators;

import hr.fer.zemris.nnao.datasets.DatasetEntry;
import hr.fer.zemris.nnao.geneticAlgorithms.Solution;
import hr.fer.zemris.nnao.neuralNetwork.INeuralNetwork;
import hr.fer.zemris.nnao.neuralNetwork.NeuralNetwork;
import hr.fer.zemris.nnao.swarmAlgorithms.AlgorithmPSO;

import java.util.List;
import java.util.function.BiFunction;
import java.util.function.Function;

import static hr.fer.zemris.nnao.neuralNetwork.NNUtil.*;

public class PSOPopulationEvaluator extends AbstractPopulationEvaluator {

    private List<DatasetEntry> trainingSet;
    private List<DatasetEntry> validationSet;
    private int populationSize;
    private int maxIterations;
    private double desiredError;
    private double desiredPrecision;
    private int maxTrys;

    private double errorFactor;
    private double weightsFactor;
    private double layersFactor;


    public PSOPopulationEvaluator(List<DatasetEntry> dataset, double trainPercentage, int populationSize,
                                  int maxIterations, double desiredError, double desiredPrecision, int maxTrys,
                                  double errorFactor, double weightsFactor, double layersFactor) {
        int index = (int) Math.round(dataset.size() * trainPercentage);
        this.trainingSet = dataset.subList(0, index);
        this.validationSet = dataset.subList(index, dataset.size());
        this.populationSize = populationSize;
        this.maxIterations = maxIterations;
        this.desiredError = desiredError;
        this.desiredPrecision = desiredPrecision;
        this.maxTrys = maxTrys;

        this.layersFactor = layersFactor;
        this.weightsFactor = weightsFactor;
        this.errorFactor = errorFactor;
    }

    @Override
    public double evaluateSolution(Solution solution) {
        INeuralNetwork nn = new NeuralNetwork(solution.getLayers(), solution.getActivations());
        double[] lowerBound = createArray(nn.getWeightsNumber(), -5.12);
        double[] upperBound = createArray(nn.getWeightsNumber(), 5.12);
        double[] lowerSpeed = createArray(nn.getWeightsNumber(), -2.);
        double[] upperSpeed = createArray(nn.getWeightsNumber(), 2.);

        BiFunction<Double, Double, Boolean> comparator = (t, u) -> Math.abs(t) > Math.abs(u);

        Function<double[], Double> particleEvaluator = createEvaluator(nn, trainingSet);
        Function<double[], Double> solutionEvaluator = createEvaluator(nn, validationSet);
        double bestFitness = Double.MAX_VALUE;
        double[] bestWeights = null;
        AlgorithmPSO pso = new AlgorithmPSO(populationSize, nn.getWeightsNumber(),
                lowerBound, upperBound, lowerSpeed, upperSpeed);

        for (int i = 0; i < maxTrys; ++i) {
            double[][] initialWeights = createInitialWeights(populationSize, solution.getLayers());
            double[] weights = pso.run(
                    particleEvaluator, comparator, desiredError, desiredPrecision, maxIterations, initialWeights
            );
            solution.setWeights(weights);
            double fitness = solutionEvaluator.apply(weights);
            if (Math.abs(fitness) < bestFitness) {
                bestFitness = Math.abs(fitness);
                bestWeights = weights;
            }
        }

        double totalFitness = layersFactor * nn.getNeuralNetworkArchitecture().length +
                weightsFactor * nn.getWeightsNumber() + errorFactor * bestFitness;

        solution.setWeights(bestWeights);
        notifyObservers(totalFitness);
        return totalFitness;
    }

    private static double[] createArray(int size, double value) {
        double[] array = new double[size];
        for (int i = 0; i < array.length; ++i) {
            array[i] = value;
        }
        return array;
    }

    private static Function<double[], Double> createEvaluator(INeuralNetwork nn, List<DatasetEntry> dataset) {
        return t -> {
            nn.setWeights(t);
            double sum = 0.;
            for (DatasetEntry d : dataset) {
                sum += Math.pow(nn.forward(d.getInput())[0] - d.getOutput()[0], 2.);
            }
            return sum / dataset.size();
        };
    }

    private static double[][] createInitialWeights(int populationSize, int[] layers) {
        double[][] initialWeights = new double[populationSize][];
        for(int j=0;j<populationSize;++j) {
            initialWeights[j] = getWeights(
                    calculateNumberOfWeights(layers), createWeightMatrices(layers)
            );
        }
        return initialWeights;
    }
}
