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

        EliminationGA ga = new EliminationGA(30, 100, 1E-3, 1E-6);
        Solution s = ga.run(
                new PopulationGenerator(3, 4, 1, 6, 2, 1),
                new SimpleCrossover(),
                new SimpleMutation(0.2, 1, 6),
                new ProportionalSelection(),
                new PSOPopulationEvaluator(dataset,40, 200,0.,1E-3)
        );
    }
}
