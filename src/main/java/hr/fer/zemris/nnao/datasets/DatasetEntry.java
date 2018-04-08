package hr.fer.zemris.nnao.datasets;

public class DatasetEntry{
    private double[] input;
    private double[] output;

    public DatasetEntry(double[] input, double[] output) {
        this.input = input;
        this.output = output;
    }

    public double[] getInput() {
        return input;
    }

    public double[] getOutput() {
        return output;
    }
}
