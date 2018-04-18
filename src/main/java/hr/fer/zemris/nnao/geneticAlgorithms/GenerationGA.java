package hr.fer.zemris.nnao.geneticAlgorithms;

import hr.fer.zemris.nnao.geneticAlgorithms.crossovers.Crossover;
import hr.fer.zemris.nnao.geneticAlgorithms.mutations.Mutation;
import hr.fer.zemris.nnao.geneticAlgorithms.selections.Selection;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class GenerationGA {

    private List<Solution> population;
    private int populationSize;

    private Solution bestSolution;

    private int currentIteration = 0;
    private int maxIterations;
    private double bestFitness = Double.MAX_VALUE;
    private double desiredFitness;
    private double desiredPrecision;

    public GenerationGA(int populationSize, int maxIterations, double desiredFitness, double desiredPrecision) {
        this.populationSize = populationSize;
        this.maxIterations = maxIterations;
        this.desiredFitness = desiredFitness;
        this.desiredPrecision = desiredPrecision;
    }

    public Solution run(IPopulationGenerator populationGenerator, Crossover crossover, Mutation mutation, Selection selection, PopulationEvaluator populationEvaluator) {

//        population = populationGenerator.createInitialPopulation(populationSize);
        fillInitialPopulation(populationGenerator,populationEvaluator);

//        for (Solution solution : population) {
//            double fitness = Math.abs(populationEvaluator.evaluateSolution(solution));
//            //abs?
//            solution.setFitness(fitness);
//            System.err.println("Fitness "+fitness);
//            if (fitness < bestFitness) {
//                bestFitness = fitness;
//                bestSolution = solution;
//            }
//        }


        while (Math.abs(bestFitness - desiredFitness) > desiredPrecision &&  currentIteration < maxIterations) {

            currentIteration++;
            double avgFitness = 0.;
            for (Solution solution : population) {
                avgFitness += solution.getFitness();
            }
            System.err.println("Average fitness: " + (avgFitness/population.size()));

            System.err.println("Iter: "+ currentIteration+ " current best fitness: " +bestFitness);

            Collections.sort(population, (s1, s2) -> (int) (s1.getFitness() - s2.getFitness()));

            Solution[] parents = selection.selectParents(population);

            List<Solution> nextGeneration = new ArrayList<>(populationSize);
            nextGeneration.add(parents[0]);
            nextGeneration.add(parents[1]);

            while(nextGeneration.size()<populationSize) {
                Solution child = crossover.doCrossover(parents[0], parents[1]);
                child = mutation.mutate(child);
                double childFitness = Math.abs(populationEvaluator.evaluateSolution(child));
                System.err.println("Child fitness: " + childFitness+ " Architecture: "+ child.toString());
                child.setFitness(childFitness);
                if (child.getFitness() < bestFitness) {
                    bestSolution = child;
                    bestFitness = child.getFitness();
                }
                nextGeneration.add(child);
            }

            population = nextGeneration;


        }

        return bestSolution;
    }

    private void fillInitialPopulation(IPopulationGenerator populationGenerator, PopulationEvaluator populationEvaluator) {
        population = new ArrayList<>(populationSize);
        int counter = 0;

        while(counter < populationSize){
            Solution solution = populationGenerator.createIndividual();
            double fitness = populationEvaluator.evaluateSolution(solution);
            if(Double.isNaN(fitness)) {
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
