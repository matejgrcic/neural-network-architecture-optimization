package hr.fer.zemris.nnao;

import hr.fer.zemris.nnao.datasets.DatasetEntry;
import hr.fer.zemris.nnao.datasets.DatasetUtils;
import hr.fer.zemris.nnao.geneticAlgorithms.AbstractGA;
import hr.fer.zemris.nnao.geneticAlgorithms.EliminationGA;
import hr.fer.zemris.nnao.geneticAlgorithms.Solution;
import hr.fer.zemris.nnao.geneticAlgorithms.crossovers.SimpleCrossover;
import hr.fer.zemris.nnao.geneticAlgorithms.evaluators.AbstractPopulationEvaluator;
import hr.fer.zemris.nnao.geneticAlgorithms.evaluators.BPPopulationEvaluator;
import hr.fer.zemris.nnao.geneticAlgorithms.evaluators.PSOPopulationEvaluator;
import hr.fer.zemris.nnao.geneticAlgorithms.generators.PopulationGenerator;
import hr.fer.zemris.nnao.geneticAlgorithms.mutations.SimpleMutation;
import hr.fer.zemris.nnao.geneticAlgorithms.selections.TournamentSelection;
import hr.fer.zemris.nnao.neuralNetwork.INeuralNetwork;
import hr.fer.zemris.nnao.neuralNetwork.NeuralNetwork;
import hr.fer.zemris.nnao.observers.evaluators.LoggerEvaluationObserver;
import hr.fer.zemris.nnao.observers.ga.ConsoleLoggerObserver;
import hr.fer.zemris.nnao.observers.ga.FileLoggerObserver;

import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.List;

public class MainIris {

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



    public static void main(String[] args) throws IOException {

        List<DatasetEntry> dataset = DatasetUtils.createIrisDataset();
        int index = (int) Math.round(dataset.size() * 0.85);
        List<DatasetEntry> trainingAndValidationDataset = dataset.subList(0,index);
        List<DatasetEntry> testDataset = dataset.subList(index,dataset.size());
        AbstractGA ga = new EliminationGA(populationSize, maxIter, desiredFitness, desiredPrecision);
        OutputStream os = Files.newOutputStream(Paths.get("./iris_graph_data.txt"), StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);

        ga.addObserver(new ConsoleLoggerObserver());

        AbstractPopulationEvaluator evaluation = new PSOPopulationEvaluator(
                trainingAndValidationDataset, 0.83,
                PSOPopulationSize, PSOMaxIter,
                desiredError, desiredPrecision,
                PSOMaxTrys, errorFactor,
                weightsFactor, layersFactor);
        evaluation.addObserver(new LoggerEvaluationObserver());


        Solution solution = ga.run(
                new PopulationGenerator(minLayersNum, maxLayersNum, minLayerSize, maxLayerSize, inputSize, outputSize),
                new SimpleCrossover(),
                new SimpleMutation(minLayerSize, maxLayerSize, minLayersNum, maxLayersNum,
                        changeLayerP, addLayerP, removeLayerP, changeActivationP),
                new TournamentSelection(numberOfSelectionCandidates, selectDuplicates),
                evaluation
        );

        os.flush();
        os.close();

        INeuralNetwork nn = new NeuralNetwork(solution.getLayers(), solution.getActivations());
        nn.setWeights(solution.getWeights());
        System.out.println("Broj pogresaka najbolje: " + calculateMisses(testDataset, nn));
    }

    private static int calculateMisses(List<DatasetEntry> dataset, INeuralNetwork nn) {
        int cnt = 0;
        for (DatasetEntry d : dataset) {
            double[] res = nn.forward(d.getInput());

            double x = Math.round(res[0]);
            double target = d.getOutput()[0];
            if (target != x) {
                cnt++;
            }
        }
        return cnt;
    }
}
