package hr.fer.zemris.nnao;

import hr.fer.zemris.nnao.datasets.DatasetEntry;
import hr.fer.zemris.nnao.datasets.DatasetUtils;
import hr.fer.zemris.nnao.geneticAlgorithms.*;
import hr.fer.zemris.nnao.geneticAlgorithms.crossovers.SimpleCrossover;
import hr.fer.zemris.nnao.geneticAlgorithms.mutations.SimpleMutation;
import hr.fer.zemris.nnao.geneticAlgorithms.selections.ProportionalSelection;
import hr.fer.zemris.nnao.geneticAlgorithms.selections.TournamentSelection;
import hr.fer.zemris.nnao.neuralNetwork.NeuralNetwork;

import java.io.IOException;
import java.util.List;

public class Main2 {

    public static void main(String[] args) throws IOException {
        List<DatasetEntry> dataset = DatasetUtils.createRastring2DDataset();

        GenerationGA ga = new GenerationGA(5, 1000, 0., 1E-3);
        Solution s = ga.run(
                new PopulationGenerator(3, 5, 6, 15, 2, 1),
                new SimpleCrossover(),
                new SimpleMutation(0.4, 1, 6),
                new TournamentSelection(4),
                new PSOPopulationEvaluator(dataset,50, 70,0.,1E-3, 3)
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



//        List<DatasetEntry> dataset = DatasetUtils.createSinXDataset();
//
//        EliminationGA ga = new EliminationGA(30, 1000, 0., 1E-3);
//        Solution s = ga.run(
//                new PopulationGenerator(3, 5, 1, 7, 1, 1),
//                new SimpleCrossover(),
//                new SimpleMutation(0.3, 1, 6),
//                new ProportionalSelection(),
//                new BPPopulationEvaluator(dataset,1E-7,1000,0.,1e-3,10,0.8)
//        );
//
//        StringBuilder sb = new StringBuilder();
//        for(int i = 0; i<s.getNumberOfLayers(); ++i){
//            sb.append(s.getArchitecture()[i]);
//            sb.append(s.getActivations()[i]+ " ");
//        }
//        System.out.println(sb.toString() + "Error: "+ s.getFitness());
    }

//    new PSOPopulationEvaluator(dataset,40, 200,0.,1E-3)
}
