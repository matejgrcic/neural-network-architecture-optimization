package hr.fer.zemris.nnao.geneticAlgorithms;

import hr.fer.zemris.nnao.geneticAlgorithms.crossovers.Crossover;
import hr.fer.zemris.nnao.geneticAlgorithms.evaluators.PopulationEvaluator;
import hr.fer.zemris.nnao.geneticAlgorithms.mutations.Mutation;
import hr.fer.zemris.nnao.geneticAlgorithms.selections.Selection;

import java.util.ArrayList;
import java.util.List;

public class GenerationGA extends AbstractGA {

    public GenerationGA(int populationSize, int maxIterations, double desiredFitness, double desiredPrecision) {
        super(populationSize, maxIterations, desiredFitness, desiredPrecision);
    }


    @Override
    protected void createNextPopulation(Selection selection, Crossover crossover, Mutation mutation, PopulationEvaluator populationEvaluator) {
        Solution[] parents = selection.selectParents(population);

        List<Solution> nextGeneration = new ArrayList<>(populationSize);
        nextGeneration.add(parents[0]);
        nextGeneration.add(parents[1]);

        while (nextGeneration.size() < populationSize) {
            Solution child = crossover.doCrossover(parents[0], parents[1]);
            child = mutation.mutate(child);
            double childFitness = Math.abs(populationEvaluator.evaluateSolution(child));
            System.err.println("Child fitness: " + childFitness + " Architecture: " + child.toString());
            child.setFitness(childFitness);

            if(Double.isNaN(childFitness)){
                continue;
            }

            if (child.getFitness() < bestFitness) {
                bestSolution = child;
                bestFitness = child.getFitness();
            }
            nextGeneration.add(child);
        }

        population = nextGeneration;
    }
}
