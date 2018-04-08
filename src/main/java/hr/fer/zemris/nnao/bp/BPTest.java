package hr.fer.zemris.nnao.bp;

import hr.fer.zemris.nnao.datasets.DatasetEntry;
import hr.fer.zemris.nnao.neuralNetwork.NeuralNetwork;
import hr.fer.zemris.nnao.neuralNetwork.activations.ActivationFunctions;
import hr.fer.zemris.nnao.neuralNetwork.activations.IActivation;

import java.util.Arrays;

public class BPTest {

    public static void main(String[] args ) {
        NeuralNetwork nn = new NeuralNetwork(
                new int[] {3, 2,1},
                new IActivation[] {ActivationFunctions.Identity, ActivationFunctions.Sigmoid, ActivationFunctions.ReLU}
        );

        double[] input = new double[] {1. ,2., 3.};
        double[] output = new double[] {2.};

        DatasetEntry entry = new DatasetEntry(input,output);

        Backpropagation bp = new Backpropagation(Arrays.asList(entry,entry,entry,entry,entry,entry, entry), Arrays.asList(entry),0.1,
                10, 0.1, 0.1 , nn, 3);
        bp.run();
    }
}
