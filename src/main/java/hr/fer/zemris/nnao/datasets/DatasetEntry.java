package hr.fer.zemris.nnao.datasets;

public class DatasetEntry<T, V> {
    private T[] input;
    private V[] output;

    public DatasetEntry(T[] input, V[] output) {
        this.input = input;
        this.output = output;
    }

    public T[] getInput() {
        return input;
    }

    public V[] getOutput() {
        return output;
    }
}
