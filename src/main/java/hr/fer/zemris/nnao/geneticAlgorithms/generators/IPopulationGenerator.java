package hr.fer.zemris.nnao.geneticAlgorithms.generators;

import hr.fer.zemris.nnao.geneticAlgorithms.Solution;

import java.util.List;

public interface IPopulationGenerator {

    List<Solution> createInitialPopulation(int populationSize);
    Solution createIndividual();
}
