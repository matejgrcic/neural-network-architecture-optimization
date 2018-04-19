package hr.fer.zemris.nnao;

import hr.fer.zemris.nnao.datasets.DatasetEntry;
import hr.fer.zemris.nnao.datasets.DatasetUtils;
import hr.fer.zemris.nnao.geneticAlgorithms.*;
import hr.fer.zemris.nnao.geneticAlgorithms.crossovers.SimpleCrossover;
import hr.fer.zemris.nnao.geneticAlgorithms.evaluators.PSOPopulationEvaluator;
import hr.fer.zemris.nnao.geneticAlgorithms.generators.PopulationGenerator;
import hr.fer.zemris.nnao.geneticAlgorithms.mutations.SimpleMutation;
import hr.fer.zemris.nnao.geneticAlgorithms.selections.TournamentSelection;
import hr.fer.zemris.nnao.neuralNetwork.NeuralNetwork;
import hr.fer.zemris.nnao.observers.ConsoleLoggerObserver;

import java.io.IOException;
import java.util.List;

public class MainSinXNormalisedGenerationGA {

    public static final int populationSize = 5;
    public static final int maxIter = 100;
    public static final int minLayersNum = 3;
    public static final int maxLayersNum = 5;
    public static final int maxLayerSize = 15;
    public static final int minLayerSize = 6;
    public static final int inputSize = 1;
    public static final int outputSize = 1;
    public static final int numberOfSelectionCandidates = 4;
    public static final double mutationProb = 0.2;
    public static final double desiredError = 0.;
    public static final double desiredFitness = 0.;
    public static final double desiredPrecision = 1E-3;
    public static final boolean selectDuplicates = false;

    public static final int PSOPopulationSize = 50;
    public static final int PSOMaxIter = 70;
    public static final int PSOMaxTrys = 2;

    public static void main(String[] args) throws IOException {
        List<DatasetEntry> dataset = DatasetUtils.createSinXDatasetNormalized();

        AbstractGA ga = new GenerationGA(populationSize, maxIter, desiredFitness, desiredPrecision);
        ga.addObserver(new ConsoleLoggerObserver());
        Solution s = ga.run(
                new PopulationGenerator(minLayersNum, maxLayersNum, minLayerSize, maxLayerSize, inputSize, outputSize),
                new SimpleCrossover(),
                new SimpleMutation(mutationProb, minLayerSize, maxLayerSize),
                new TournamentSelection(numberOfSelectionCandidates, selectDuplicates),
                new PSOPopulationEvaluator(dataset, PSOPopulationSize, PSOMaxIter, desiredError, desiredPrecision, PSOMaxTrys)
        );

        StringBuilder sb = new StringBuilder();
        for(int i = 0; i<s.getNumberOfLayers(); ++i){
            sb.append(s.getArchitecture()[i]);
            sb.append(s.getActivations()[i]+ " ");
        }
        System.out.println(sb.toString() + "Error: "+ s.getFitness());

        NeuralNetwork nn = new NeuralNetwork(s.getArchitecture(),s.getActivations());
        nn.setWeights(s.getWeights());

        double sum = 0.;
        for (DatasetEntry d : dataset) {
            sum += Math.pow(nn.forward(d.getInput())[0] - d.getOutput()[0],2.);
        }
        System.out.println(sum/dataset.size());
    }
}
