package hr.fer.zemris.nnao.observers.algorithms;

import hr.fer.zemris.nnao.geneticAlgorithms.AbstractGA;

public class ConsoleLoggerObserver implements GAObserver {


    private StringBuilder logger = new StringBuilder();

    @Override
    public void update(AbstractGA geneticAlgorithm) {
        logger.setLength(0);

        logger.append("#Iteration: "+geneticAlgorithm.getCurrentIteration()+"\n");
        logger.append("\tBest fitness: "+geneticAlgorithm.getBestFitness()+"\n");
        logger.append("\tBest architecture: "+geneticAlgorithm.getBestSolution().toString()+"\n");
        logger.append("\tAverage population fitness: "+geneticAlgorithm.getAverageFitness()+"\n");

        System.out.print(logger.toString());
    }
}
