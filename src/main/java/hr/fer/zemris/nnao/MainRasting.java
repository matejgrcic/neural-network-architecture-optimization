package hr.fer.zemris.nnao;

import hr.fer.zemris.nnao.datasets.DatasetEntry;
import hr.fer.zemris.nnao.datasets.DatasetUtils;
import hr.fer.zemris.nnao.geneticAlgorithms.AbstractGA;
import hr.fer.zemris.nnao.geneticAlgorithms.EliminationGA;
import hr.fer.zemris.nnao.geneticAlgorithms.Solution;
import hr.fer.zemris.nnao.geneticAlgorithms.crossovers.SimpleCrossover;
import hr.fer.zemris.nnao.geneticAlgorithms.evaluators.AbstractPopulationEvaluator;
import hr.fer.zemris.nnao.geneticAlgorithms.evaluators.PSOPopulationEvaluator;
import hr.fer.zemris.nnao.geneticAlgorithms.generators.PopulationGenerator;
import hr.fer.zemris.nnao.geneticAlgorithms.mutations.SimpleMutation;
import hr.fer.zemris.nnao.geneticAlgorithms.selections.TournamentSelection;
import hr.fer.zemris.nnao.neuralNetwork.INeuralNetwork;
import hr.fer.zemris.nnao.neuralNetwork.NeuralNetwork;
import hr.fer.zemris.nnao.observers.evaluators.LoggerEvaluationObserver;
import hr.fer.zemris.nnao.observers.ga.ConsoleLoggerObserver;

import java.io.IOException;
import java.util.List;

public class MainRasting {

    public static final double solutionDelta = 0.01;
    public static final int populationSize = 10;
    public static final int maxIter = 70;
    public static final int minLayersNum = 3;
    public static final int maxLayersNum = 5;
    public static final int maxLayerSize = 130;
    public static final int minLayerSize = 50;
    public static final int inputSize = 2;
    public static final int outputSize = 1;
    public static final int numberOfSelectionCandidates = 4;
    public static final double mutationProb = 0.05;
    public static final double desiredError = 0.;
    public static final double desiredFitness = 0.;
    public static final double desiredPrecision = 1E-3;
    public static final boolean selectDuplicates = false;

    public static final double weightsFactor = 1E-4;
    public static final double layersFactor = 1E-2;
    public static final double errorFactor = 1.;
    public static final double addLayerP = 0.07;
    public static final double removeLayerP = 0.07;
    public static final double changeLayerP = 0.2;
    public static final double changeActivationP = 0.2;

    public static void main(String[] args) throws IOException {

        List<DatasetEntry> dataset = DatasetUtils.createRastring2DDataset();

        AbstractGA ga = new EliminationGA(populationSize, maxIter, desiredFitness, desiredPrecision);

        ga.addObserver(new ConsoleLoggerObserver());

        AbstractPopulationEvaluator evaluator = new PSOPopulationEvaluator(dataset, 0.8, 50, 100, desiredError, desiredPrecision, 1,
                errorFactor, weightsFactor, layersFactor);
        evaluator.addObserver(new LoggerEvaluationObserver());

        Solution s = ga.run(
                new PopulationGenerator(minLayersNum, maxLayersNum, minLayerSize, maxLayerSize, inputSize, outputSize),
                new SimpleCrossover(),
                new SimpleMutation(minLayerSize, maxLayerSize, minLayersNum, maxLayersNum,
                        changeLayerP, addLayerP, removeLayerP, changeActivationP),
                new TournamentSelection(numberOfSelectionCandidates, selectDuplicates),
//                new BPPopulationEvaluator(dataset, learningRate, maxIterBP, desiredError, desiredPrecision, batchSize, trainPercentage)
                evaluator
        );

        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < s.getNumberOfLayers(); ++i) {
            sb.append(s.getLayers()[i]);
            sb.append(s.getActivations()[i] + " ");
        }
        System.out.println(sb.toString() + "Error: " + s.getFitness());

        INeuralNetwork nn = new NeuralNetwork(s.getLayers(), s.getActivations());
        nn.setWeights(s.getWeights());

        double sum = 0.;
        for (DatasetEntry d : dataset) {
            sum += Math.pow(nn.forward(d.getInput())[0] - d.getOutput()[0], 2.);
        }
        System.out.println(sum / dataset.size());


    }
}
