package hr.fer.zemris.nnao.geneticAlgorithms;

import java.util.List;

public interface IPopulationGenerator {

    List<Solution> createInitialPopulation(int populationSize);
}
