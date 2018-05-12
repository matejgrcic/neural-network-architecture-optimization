package hr.fer.zemris.nnao.neuralNetwork;

import hr.fer.zemris.nnao.neuralNetwork.activations.ActivationFunctions;
import hr.fer.zemris.nnao.neuralNetwork.activations.IActivation;

public class NNtest {

    public static void main(String[] args ) {
        INeuralNetwork nn = new NeuralNetwork(
                new int[] {3, 2,1},
                new IActivation[] {ActivationFunctions.Identity, ActivationFunctions.Sigmoid, ActivationFunctions.ReLU}
                );

        double[] input = new double[] {1. ,2., 3.};
        double[] res = nn.forward(input);
    }
}
