package hr.fer.zemris.nnao.geneticAlgorithms.mutations;

import hr.fer.zemris.nnao.geneticAlgorithms.Solution;

public interface Mutation {

    Solution mutate(Solution solution);
}
