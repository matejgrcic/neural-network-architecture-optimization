package hr.fer.zemris.nnao.geneticAlgorithms;

import hr.fer.zemris.nnao.geneticAlgorithms.crossovers.Crossover;
import hr.fer.zemris.nnao.geneticAlgorithms.evaluators.PopulationEvaluator;
import hr.fer.zemris.nnao.geneticAlgorithms.mutations.Mutation;
import hr.fer.zemris.nnao.geneticAlgorithms.selections.Selection;

public class EliminationGA extends AbstractGA {


    public EliminationGA(int populationSize, int maxIterations, double desiredFitness, double desiredPrecision, double solutionDelta) {
        super(populationSize, maxIterations, desiredFitness, desiredPrecision, solutionDelta);
    }


    @Override
    protected void createNextPopulation(Selection selection, Crossover crossover, Mutation mutation, PopulationEvaluator populationEvaluator) {

        Solution[] parents = selection.selectParents(population);
        Solution child = crossover.doCrossover(parents[0], parents[1]);
        child = mutation.mutate(child);
        double childFitness = Math.abs(populationEvaluator.evaluateSolution(child));
        child.setFitness(childFitness);

        if (child.getFitness() < population.get(populationSize - 1).getFitness()) {
            population.remove(populationSize - 1);
            population.add(child);
        }

        if (child.getFitness() < bestFitness) {
            bestSolution = child;
            bestFitness = child.getFitness();
        }
    }
}
