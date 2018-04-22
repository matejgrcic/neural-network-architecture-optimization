package hr.fer.zemris.nnao;

import hr.fer.zemris.nnao.datasets.DatasetEntry;
import hr.fer.zemris.nnao.datasets.DatasetUtils;
import hr.fer.zemris.nnao.geneticAlgorithms.AbstractGA;
import hr.fer.zemris.nnao.geneticAlgorithms.GenerationGA;
import hr.fer.zemris.nnao.geneticAlgorithms.Solution;
import hr.fer.zemris.nnao.geneticAlgorithms.crossovers.SimpleCrossover;
import hr.fer.zemris.nnao.geneticAlgorithms.evaluators.AbstractPopulationEvaluator;
import hr.fer.zemris.nnao.geneticAlgorithms.evaluators.PSOPopulationEvaluator;
import hr.fer.zemris.nnao.geneticAlgorithms.generators.PopulationGenerator;
import hr.fer.zemris.nnao.geneticAlgorithms.mutations.SimpleMutation;
import hr.fer.zemris.nnao.geneticAlgorithms.selections.TournamentSelection;
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

public class MainIrisGeneration {

    public static final int populationSize = 10;
    public static final int maxIter = 5;
    public static final int minLayersNum = 3;
    public static final int maxLayersNum = 5;
    public static final int maxLayerSize = 110;
    public static final int minLayerSize = 80;
    public static final int inputSize = 4;
    public static final int outputSize = 1;
    public static final int numberOfSelectionCandidates = 4;
    public static final double mutationProb = 0.05;
    public static final double desiredError = 0.;
    public static final double desiredFitness = 0.;
    public static final double desiredPrecision = 1e-5;
    public static final boolean selectDuplicates = false;

    public static final double learningRate = 1E-4;
    public static final double trainPercentage = 0.7;
    public static final int batchSize = 30;
    public static final int maxIterBP = 50_000;

    public static void main(String[] args) throws IOException {

        List<DatasetEntry> dataset = DatasetUtils.createIrisDataset();

        AbstractGA ga = new GenerationGA(populationSize, maxIter, desiredFitness, desiredPrecision);

        OutputStream os = Files.newOutputStream(Paths.get("./iris_result.txt"), StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);


        ga.addObserver(new ConsoleLoggerObserver());
        ga.addObserver(new FileLoggerObserver(new BufferedOutputStream(os)));

        AbstractPopulationEvaluator evaluation = new PSOPopulationEvaluator(dataset,50,100,desiredError,desiredPrecision,1);
        //        AbstractPopulationEvaluator evaluation = new BPPopulationEvaluator(dataset,learningRate,maxIterBP,desiredError,desiredPrecision,batchSize,trainPercentage);
        evaluation.addObserver(new LoggerEvaluationObserver());
        Solution s = ga.run(
                new PopulationGenerator(minLayersNum, maxLayersNum, minLayerSize, maxLayerSize, inputSize, outputSize),
                new SimpleCrossover(),
                new SimpleMutation(mutationProb, minLayerSize, maxLayerSize),
                new TournamentSelection(numberOfSelectionCandidates, selectDuplicates),
                evaluation
        );

        os.close();

        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < s.getNumberOfLayers(); ++i) {
            sb.append(s.getArchitecture()[i]);
            sb.append(s.getActivations()[i] + " ");
        }
        System.out.println(sb.toString() + "Error: " + s.getFitness());

        NeuralNetwork nn = new NeuralNetwork(s.getArchitecture(), s.getActivations());
        nn.setWeights(s.getWeights());

        int cnt = 0;
        double sum = 0.;
        for (DatasetEntry d : dataset) {
            double[] res = nn.forward(d.getInput());

            double x = Math.round(res[0]);
            double target = d.getOutput()[0];
            System.out.println("Expected: "+target + " Dobiveno: "+ x);
            if(target != x) {
                cnt++;
            }


        }
        System.out.println("Falilo: "+cnt);

        for(double x : s.getWeights()){
            System.out.println(x);
        }
    }
}
