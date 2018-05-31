package hr.fer.zemris.nnao.geneticAlgorithms;

import hr.fer.zemris.nnao.geneticAlgorithms.crossovers.Crossover;
import hr.fer.zemris.nnao.geneticAlgorithms.evaluators.PopulationEvaluator;
import hr.fer.zemris.nnao.geneticAlgorithms.generators.IPopulationGenerator;
import hr.fer.zemris.nnao.geneticAlgorithms.mutations.Mutation;
import hr.fer.zemris.nnao.geneticAlgorithms.selections.Selection;
import hr.fer.zemris.nnao.observers.ga.GAObserver;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import static hr.fer.zemris.nnao.geneticAlgorithms.GAUtil.solutionComparator;

public abstract class AbstractGA {

    protected List<Solution> population;
    protected int populationSize;
    protected Solution bestSolution;

    protected int currentIteration;
    protected int maxIterations;

    protected double bestFitness = Double.MAX_VALUE;
    protected double averageFitness = Double.MAX_VALUE;
    private double desiredFitness;
    private double desiredPrecision;

    private List<GAObserver> observers = new ArrayList<>();

    public AbstractGA(int populationSize, int maxIterations, double desiredFitness, double desiredPrecision) {
        this.populationSize = populationSize;
        this.maxIterations = maxIterations;
        this.desiredFitness = desiredFitness;
        this.desiredPrecision = desiredPrecision;
    }

    public Solution run(IPopulationGenerator populationGenerator, Crossover crossover, Mutation mutation,
                        Selection selection, PopulationEvaluator populationEvaluator) {

        fillInitialPopulation(populationGenerator, populationEvaluator);
        calculateAverageFitness();
        notifyObservers();

        while (Math.abs(bestFitness - desiredFitness) > desiredPrecision && currentIteration < maxIterations) {
            currentIteration++;
            calculateAverageFitness();
            Collections.sort(population, solutionComparator);
            createNextPopulation(selection, crossover, mutation, populationEvaluator);
            notifyObservers();
        }

        Collections.sort(population, solutionComparator);
        return bestSolution;
    }

    protected abstract void createNextPopulation(Selection selection, Crossover crossover,
                                                 Mutation mutation, PopulationEvaluator populationEvaluator);

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

    public void addObserver(GAObserver observer) {
        observers.add(observer);
    }

    public void removeObserver(GAObserver observer) {
        observers.remove(observer);
    }

    public void notifyObservers() {
        new ArrayList<>(observers).forEach(t -> t.update(this));
    }

    public int getPopulationSize() {
        return populationSize;
    }

    public Solution getBestSolution() {
        return bestSolution;
    }

    public int getCurrentIteration() {
        return currentIteration;
    }

    public int getMaxIterations() {
        return maxIterations;
    }

    public double getBestFitness() {
        return bestFitness;
    }

    public double getAverageFitness() {
        return averageFitness;
    }

    public List<Solution> getPopulation() {
        return new ArrayList<>(population);
    }
}
