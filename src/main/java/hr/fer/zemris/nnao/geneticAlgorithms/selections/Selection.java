package hr.fer.zemris.nnao.geneticAlgorithms.selections;

import hr.fer.zemris.nnao.geneticAlgorithms.Solution;

import java.util.List;

public interface Selection {

    Solution[] selectParents(List<Solution> population);
}
