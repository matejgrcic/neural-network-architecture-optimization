package hr.fer.zemris.nnao.swarmAlgorithms;

import java.util.Random;
import java.util.function.BiFunction;
import java.util.function.Function;

public class AlgorithmPSO {

    public static Random rand = new Random();
    public static double C1 = 2.;
    public static double C2 = 2.;

    private int populationSize;
    private int solutionSize;
    private double[] minPosition;
    private double[] maxPosition;
    private double[] minVelocity;
    private double[] maxVelocity;

    public AlgorithmPSO(int populationSize, int solutionSize, double[] minPosition, double[] maxPosition, double[] minVelocity, double[] maxVelocity) {
        this.populationSize = populationSize;
        this.solutionSize = solutionSize;
        this.minPosition = minPosition;
        this.maxPosition = maxPosition;
        this.minVelocity = minVelocity;
        this.maxVelocity = maxVelocity;
    }

    public double[] run(Function<double[], Double> particleEvaluator,
                        BiFunction<Double, Double, Boolean> valueComparator,
                        double desiredError, double desiredPrecision, long maxIter) {
        double[][] particlesPositions = initializePopulation(minPosition, maxPosition);
        double[][] particlesVelocities = initializePopulation(minVelocity, maxVelocity);
        double[] particleValue = new double[populationSize];

        double[] particleBestValue = new double[populationSize];
        for (int i = 0; i < populationSize; ++i) {
            particleBestValue[i] = -Double.MAX_VALUE;
        }
        double[][] particleBestPosition = new double[populationSize][solutionSize];

        double globalBestValue = -Double.MAX_VALUE;
        double[] globalBestPosition = new double[solutionSize];

        for (long iter = 0; iter < maxIter; ++iter) {

            for (int i = 0; i < populationSize; ++i) {
                particleValue[i] = particleEvaluator.apply(particlesPositions[i]);
            }

            for (int i = 0; i < populationSize; ++i) {
                if (valueComparator.apply(particleBestValue[i], particleValue[i])) {
                    particleBestValue[i] = particleValue[i];
                    particleBestPosition[i] = particlesPositions[i];
                }
            }

            for (int i = 0; i < populationSize; ++i) {
                if (valueComparator.apply(globalBestValue, particleValue[i])) {
                    globalBestValue = particleValue[i];
                    globalBestPosition = particlesPositions[i];
                }
            }

            System.err.println("Iter " + (iter + 1) + " GBV: " + String.format("%.2f", globalBestValue));
            if (Math.abs(globalBestValue - desiredError) < desiredPrecision) {
                break;
            }

            for (int i = 0; i < populationSize; ++i) {
                for (int j = 0; j < solutionSize; ++j) {
                    particlesVelocities[i][j] += C1 * rand.nextDouble() * (particleBestPosition[i][j] - particlesPositions[i][j])
                            + C2 * rand.nextDouble() * (globalBestPosition[j] - particlesPositions[i][j]);
                    particlesVelocities[i][j] = setValueInRange(particlesVelocities[i][j], minVelocity[j], maxVelocity[j]);
                    particlesPositions[i][j] += particlesVelocities[i][j];
                    particlesPositions[i][j] = setValueInRange(particlesVelocities[i][j], minPosition[j], maxPosition[j]);
                }
            }

        }

        return globalBestPosition;
    }

    private double setValueInRange(double value, double minRange, double maxRange) {
        value = Math.min(value, maxRange);
        value = Math.max(value, minRange);
        return value;
    }


    private double[][] initializePopulation(double[] minValue, double[] maxValue) {
        double[][] array = new double[populationSize][solutionSize];
        for (int i = 0; i < populationSize; ++i) {
            for (int j = 0; j < solutionSize; ++j) {
                array[i][j] = minValue[j] + (maxValue[j] - minValue[j]) * rand.nextDouble();
            }
        }
        return array;
    }

}
