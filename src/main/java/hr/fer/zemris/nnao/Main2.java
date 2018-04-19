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

import java.io.IOException;
import java.util.List;

public class Main2 {

    public static final int populationSize = 10;
    public static final int maxIter = 1000;
    public static final int minLayersNum = 3;
    public static final int maxLayersNum = 5;
    public static final int maxLayerSize = 15;
    public static final int minLayerSize = 6;
    public static final int inputSize = 2;
    public static final int outputSize = 1;
    public static final int numberOfSelectionCandidates = 4;
    public static final double mutationProb = 0.2;
    public static final double desiredError = 0.;
    public static final double desiredFitness = 0.;
    public static final double desiredPrecision = 1E-3;

    public static final int PSOPopulationSize = 50;
    public static final int PSOMaxIter = 70;
    public static final int PSOMaxTrys = 3;

    public static void main(String[] args) throws IOException {
        List<DatasetEntry> dataset = DatasetUtils.createRastring2DDataset();

        AbstractGA ga = new GenerationGA(populationSize, maxIter, desiredFitness, desiredPrecision);
        Solution s = ga.run(
                new PopulationGenerator(minLayersNum, maxLayersNum, minLayerSize, maxLayerSize, inputSize, outputSize),
                new SimpleCrossover(),
                new SimpleMutation(mutationProb, minLayerSize, maxLayerSize),
                new TournamentSelection(numberOfSelectionCandidates),
                new PSOPopulationEvaluator(dataset, PSOPopulationSize, PSOMaxIter, desiredError, desiredPrecision, PSOMaxTrys)
//                new BPPopulationEvaluator(dataset,1E-5,50_000,0.,1e-3,30,0.9)
        );

//        new BPPopulationEvaluator(dataset,1E-7,10_000,0.,1e-3,10,0.8)

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

//    new PSOPopulationEvaluator(dataset,40, 200,0.,1E-3)
}
