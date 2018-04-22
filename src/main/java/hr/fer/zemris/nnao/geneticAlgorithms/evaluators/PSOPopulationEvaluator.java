package hr.fer.zemris.nnao.geneticAlgorithms.evaluators;

import hr.fer.zemris.nnao.datasets.DatasetEntry;
import hr.fer.zemris.nnao.geneticAlgorithms.Solution;
import hr.fer.zemris.nnao.neuralNetwork.NeuralNetwork;
import hr.fer.zemris.nnao.swarmAlgorithms.AlgorithmPSO;

import java.util.List;
import java.util.function.BiFunction;
import java.util.function.Function;

public class PSOPopulationEvaluator extends AbstractPopulationEvaluator {

    private List<DatasetEntry> dataset;
    private int populationSize;
    private int maxIterations;
    private double desiredError;
    private double desiredPrecision;
    private int maxTrys;

    public PSOPopulationEvaluator(List<DatasetEntry> dataset, int populationSize, int maxIterations, double desiredError, double desiredPrecision, int maxTrys) {
        this.dataset = dataset;
        this.populationSize = populationSize;
        this.maxIterations = maxIterations;
        this.desiredError = desiredError;
        this.desiredPrecision = desiredPrecision;
        this.maxTrys = maxTrys;
    }

    @Override
    public double evaluateSolution(Solution solution) {
        NeuralNetwork nn = new NeuralNetwork(solution.getArchitecture(), solution.getActivations());
        nn.setWeights(solution.getWeights());
        double[] lowerBound = new double[nn.getWeightsNumber()];
        for (int i = 0; i < lowerBound.length; ++i) {
            lowerBound[i] = -5.12;
        }
        double[] upperBound = new double[nn.getWeightsNumber()];
        for (int i = 0; i < lowerBound.length; ++i) {
            upperBound[i] = 5.12;
        }
        double[] lowerSpeed = new double[nn.getWeightsNumber()];
        for (int i = 0; i < lowerBound.length; ++i) {
            lowerSpeed[i] = -2.;
        }
        double[] upperSpeed = new double[nn.getWeightsNumber()];
        for (int i = 0; i < lowerBound.length; ++i) {
            upperSpeed[i] = 2.;
        }

        BiFunction<Double, Double, Boolean> comparator = (t, u) -> Math.abs(t) > Math.abs(u);
        Function<double[], Double> particleEvaluator = t -> {
            nn.setWeights(t);
            double sum = 0.;
            for (DatasetEntry d : dataset) {
                sum += Math.pow(nn.forward(d.getInput())[0] - d.getOutput()[0], 2.);
            }
            return sum / dataset.size();
        };

        double bestFitness = Double.MAX_VALUE;
        for(int i = 0; i<maxTrys; ++i) {
            AlgorithmPSO pso = new AlgorithmPSO(populationSize, nn.getWeightsNumber(), lowerBound, upperBound, lowerSpeed, upperSpeed);
            double[] result = pso.run(particleEvaluator, comparator, desiredError, desiredPrecision, maxIterations);
            solution.setWeights(result);
            double fitness = particleEvaluator.apply(result);
            if (Math.abs(fitness) < bestFitness) {
                bestFitness = Math.abs(fitness);
            }
        }
        notifyObservers(bestFitness);
        return bestFitness;
    }
}
