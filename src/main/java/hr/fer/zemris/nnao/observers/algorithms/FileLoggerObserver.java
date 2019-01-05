package hr.fer.zemris.nnao.observers.algorithms;

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
        logger.delete(0, logger.length());

        logger.append(geneticAlgorithm.getBestFitness() + "," + geneticAlgorithm.getAverageFitness()+"\n");

        try {
            outputStream.write(logger.toString().getBytes());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
