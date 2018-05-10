package hr.fer.zemris.nnao.observers.ga;

import hr.fer.zemris.nnao.geneticAlgorithms.AbstractGA;

import java.io.IOException;
import java.io.OutputStream;

public class GraphDataObserver implements GAObserver {

    private OutputStream outputStream;
    private int iteration;

    public GraphDataObserver(OutputStream outputStream) {
        this.outputStream = outputStream;
    }

    @Override
    public void update(AbstractGA geneticAlgorithm) {
        StringBuilder sb = new StringBuilder();
        sb.append(iteration+",");
        sb.append(geneticAlgorithm.getBestFitness()+",");
        sb.append(geneticAlgorithm.getAverageFitness()+"\n");
        try {
            outputStream.write(sb.toString().getBytes());
            outputStream.flush();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
