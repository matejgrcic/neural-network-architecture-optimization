package hr.fer.zemris.nnao.geneticAlgorithms.evaluators;

import hr.fer.zemris.nnao.geneticAlgorithms.Solution;

public interface PopulationEvaluator {

    double evaluateSolution(Solution solution);
}
