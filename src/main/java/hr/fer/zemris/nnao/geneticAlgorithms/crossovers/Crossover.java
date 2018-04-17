package hr.fer.zemris.nnao.geneticAlgorithms.crossovers;

import hr.fer.zemris.nnao.geneticAlgorithms.Solution;

public interface Crossover {

    Solution doCrossover(Solution first, Solution second);
}
