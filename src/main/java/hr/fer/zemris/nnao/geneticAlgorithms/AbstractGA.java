package hr.fer.zemris.nnao.geneticAlgorithms;

import hr.fer.zemris.nnao.geneticAlgorithms.crossovers.Crossover;
import hr.fer.zemris.nnao.geneticAlgorithms.evaluators.PopulationEvaluator;
import hr.fer.zemris.nnao.geneticAlgorithms.generators.IPopulationGenerator;
import hr.fer.zemris.nnao.geneticAlgorithms.mutations.Mutation;
import hr.fer.zemris.nnao.geneticAlgorithms.selections.Selection;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public abstract class AbstractGA {

    protected List<Solution> population;
    protected int populationSize;
    protected Solution bestSolution;

    protected int currentIteration = 0;
    protected int maxIterations;

    protected double bestFitness = Double.MAX_VALUE;
    protected double averageFitness = Double.MAX_VALUE;

    private double desiredFitness;
    private double desiredPrecision;

    public AbstractGA(int populationSize, int maxIterations, double desiredFitness, double desiredPrecision) {
        this.populationSize = populationSize;
        this.maxIterations = maxIterations;
        this.desiredFitness = desiredFitness;
        this.desiredPrecision = desiredPrecision;
    }

    public Solution run(IPopulationGenerator populationGenerator, Crossover crossover, Mutation mutation, Selection selection, PopulationEvaluator populationEvaluator) {

        fillInitialPopulation(populationGenerator, populationEvaluator);

        while (Math.abs(bestFitness - desiredFitness) > desiredPrecision && currentIteration < maxIterations) {

            currentIteration++;

            calculateAverageFitness();
            System.err.println("Average fitness: " + averageFitness);

            System.err.println("Iter: " + currentIteration + " current best fitness: " + bestFitness);

            Collections.sort(population, (s1, s2) -> (int) (s1.getFitness() - s2.getFitness()));

            createNextPopulation(selection, crossover, mutation, populationEvaluator);
        }

        return bestSolution;
    }

    //treba napraviti novu populaciju i vratiti bes
    protected abstract void createNextPopulation(Selection selection, Crossover crossover, Mutation mutation, PopulationEvaluator populationEvaluator);

    protected void calculateAverageFitness() {
        double total = 0.;
        for (Solution solution : population) {
            total += solution.getFitness();
        }
        averageFitness = total / populationSize;
    }

    protected void fillInitialPopulation(IPopulationGenerator populationGenerator, PopulationEvaluator populationEvaluator) {
        population = new ArrayList<>(populationSize);
        int counter = 0;

        while (counter < populationSize) {
            Solution solution = populationGenerator.createIndividual();
            double fitness = populationEvaluator.evaluateSolution(solution);
            if (Double.isNaN(fitness)) {
                continue;
            }
            solution.setFitness(fitness);
            population.add(solution);

            if (fitness < bestFitness) {
                bestFitness = fitness;
                bestSolution = solution;
            }
            counter++;
        }
    }
}