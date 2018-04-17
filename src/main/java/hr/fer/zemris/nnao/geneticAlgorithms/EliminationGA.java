package hr.fer.zemris.nnao.geneticAlgorithms;

import hr.fer.zemris.nnao.geneticAlgorithms.crossovers.Crossover;
import hr.fer.zemris.nnao.geneticAlgorithms.mutations.Mutation;
import hr.fer.zemris.nnao.geneticAlgorithms.selections.Selection;

import java.util.Collections;
import java.util.List;

public class EliminationGA {

    private List<Solution> population;
    private int populationSize;

    private Solution bestSolution;

    private int currentIteration = 0;
    private int maxIterations;
    private double bestFitness = -Double.MAX_VALUE;
    private double desiredFitness;
    private double desiredPrecision;

    public EliminationGA(int populationSize, int maxIterations, double desiredFitness, double desiredPrecision) {
        this.populationSize = populationSize;
        this.maxIterations = maxIterations;
        this.desiredFitness = desiredFitness;
        this.desiredPrecision = desiredPrecision;
    }

    public Solution run(IPopulationGenerator populationGenerator, Crossover crossover, Mutation mutation, Selection selection, PopulationEvaluator populationEvaluator) {

        population = populationGenerator.createInitialPopulation(populationSize);

        for (Solution solution : population) {
            double fitness = populationEvaluator.evaluateSolution(solution);

            //abs?
            solution.setFitness(Math.abs(fitness));

            if (fitness < bestFitness) {
                bestFitness = fitness;
                bestSolution = solution;
            }
        }


        while (Math.abs(bestFitness - desiredFitness) > desiredPrecision &&  currentIteration < maxIterations) {

            currentIteration++;

            System.err.println(bestFitness);

            Collections.sort(population, (s1, s2) -> (int) (s1.getFitness() - s2.getFitness()));

            Solution[] parents = selection.selectParents(population);
            Solution child = crossover.doCrossover(parents[0], parents[1]);
            child = mutation.mutate(child);
            double childFitness = Math.abs(populationEvaluator.evaluateSolution(child));
            child.setFitness(childFitness);

            if (population.get(populationSize - 1).getFitness() > child.getFitness()) {
                population.remove(populationSize - 1);
                population.add(child);
            }

            if (child.getFitness() < bestFitness) {
                bestSolution = child;
                bestFitness = child.getFitness();
            }
        }


        return bestSolution;
    }
}
