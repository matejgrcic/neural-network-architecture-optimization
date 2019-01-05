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

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class DemoIris {

    public static final int populationSize = 12;
    public static final int maxIter = 70;
    public static final int minLayersNum = 3;
    public static final int maxLayersNum = 5;
    public static final int maxLayerSize = 100;
    public static final int minLayerSize = 40;
    public static final int inputSize = 4;
    public static final int outputSize = 1;
    public static final int numberOfSelectionCandidates = 4;
    public static final double desiredError = 0.;
    public static final double desiredFitness = 0.;
    public static final double desiredPrecision = 1e-5;
    public static final boolean selectDuplicates = false;

    public static final double weightsFactor = 1E-5;
    public static final double layersFactor = 1E-3;
    public static final double errorFactor = 1.;
    public static final double addLayerP = 0.4;
    public static final double removeLayerP = 0.4;
    public static final double changeLayerP = 0.4;
    public static final double changeActivationP = 0.4;

    public static final int PSOPopulationSize = 50;
    public static final int PSOMaxIter = 150;
    public static final int PSOMaxTrys = 5;

    public static final double trainPercentage = 0.83;

    public static void main(String[] args) throws IOException {
        var datasets = DemoIris.testTrainSplit(0.85);
        List<DatasetEntry> trainingAndValidationDataset = datasets.get(0);
        List<DatasetEntry> testDataset = datasets.get(1);

        AbstractGA algorithm = DemoIris.createGeneticAlgorithm();
        AbstractPopulationEvaluator evaluation = DemoIris.createPopulationEvaluator(
            trainingAndValidationDataset, trainPercentage
        );

        Solution solution = algorithm.run(
                new PopulationGenerator(
                    minLayersNum, maxLayersNum,
                    minLayerSize, maxLayerSize,
                    inputSize, outputSize
                ),
                new SimpleCrossover(),
                new SimpleMutation(
                    minLayerSize, maxLayerSize,
                    minLayersNum, maxLayersNum,
                    changeLayerP, addLayerP,
                    removeLayerP, changeActivationP
                ),
                new TournamentSelection(numberOfSelectionCandidates, selectDuplicates),
                evaluation
        );

        INeuralNetwork neuralNetwork = new NeuralNetwork(solution.getLayers(), solution.getActivations());
    }

    private static AbstractGA createGeneticAlgorithm() {
        return new EliminationGA(populationSize, maxIter, desiredFitness, desiredPrecision);
    }

    private static AbstractPopulationEvaluator createPopulationEvaluator(
        List<DatasetEntry> trainingAndValidationDataset,
        double trainPercentage) {
        return new PSOPopulationEvaluator(
            trainingAndValidationDataset, trainPercentage,
            PSOPopulationSize, PSOMaxIter,
            desiredError, desiredPrecision,
            PSOMaxTrys, errorFactor,
            weightsFactor, layersFactor);
    }

    private static List<List<DatasetEntry>> testTrainSplit(double percent) throws IOException {
        List<DatasetEntry> dataset = DatasetUtils.createIrisDataset();
        var index = (int) Math.round(dataset.size() * percent);
        var datasets = new ArrayList<List<DatasetEntry>>();
        datasets.add(dataset.subList(0,index));
        datasets.add(dataset.subList(index,dataset.size()));
        return datasets;
    }
}
