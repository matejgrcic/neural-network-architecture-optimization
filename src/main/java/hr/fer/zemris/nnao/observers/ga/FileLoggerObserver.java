package hr.fer.zemris.nnao.observers.ga;

import hr.fer.zemris.nnao.geneticAlgorithms.AbstractGA;

import java.io.IOException;
import java.io.OutputStream;

public class FileLoggerObserver implements GAObserver {

    private StringBuilder logger = new StringBuilder();
    private OutputStream outputStream;

    public FileLoggerObserver(OutputStream outputStream) {
        this.outputStream = outputStream;
    }


    @Override
    public void update(AbstractGA geneticAlgorithm) {
        logger.setLength(0);

        logger.append("#Iteration: "+geneticAlgorithm.getCurrentIteration()+"\n");
        logger.append("\tBest fitness: "+geneticAlgorithm.getBestFitness()+"\n");
        logger.append("\tBest architecture: "+geneticAlgorithm.getBestSolution().toString()+"\n");
        logger.append("\tAverage population fitness: "+geneticAlgorithm.getAverageFitness()+"\n");

        try {
            outputStream.write(logger.toString().getBytes());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
