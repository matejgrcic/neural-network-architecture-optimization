package hr.fer.zemris.nnao;

import hr.fer.zemris.nnao.datasets.DatasetEntry;
import hr.fer.zemris.nnao.datasets.DatasetUtils;
import hr.fer.zemris.nnao.geneticAlgorithms.*;
import hr.fer.zemris.nnao.geneticAlgorithms.crossovers.SimpleCrossover;
import hr.fer.zemris.nnao.geneticAlgorithms.mutations.SimpleMutation;
import hr.fer.zemris.nnao.geneticAlgorithms.selections.ProportionalSelection;

import java.io.IOException;
import java.util.List;

public class Main {

    public static void main(String[] args) throws IOException {
        List<DatasetEntry> dataset = DatasetUtils.createRastring2DDataset();

        EliminationGA ga = new EliminationGA(3, 20, 1E-3, 1E-6);
        Solution s = ga.run(
                new PopulationGenerator(3, 5, 1, 8, 2, 1),
                new SimpleCrossover(),
                new SimpleMutation(0.2, 1, 8),
                new ProportionalSelection(),
                new BPPopulationEvaluator(dataset, 0.2, 10000, 1E-3, 1E-6, 32, 0.8)
        );


    }
}
