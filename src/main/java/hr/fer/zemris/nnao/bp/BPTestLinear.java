package hr.fer.zemris.nnao.bp;

import hr.fer.zemris.nnao.datasets.DatasetEntry;
import hr.fer.zemris.nnao.datasets.DatasetUtils;
import hr.fer.zemris.nnao.neuralNetwork.NeuralNetwork;
import hr.fer.zemris.nnao.neuralNetwork.activations.ActivationFunctions;
import hr.fer.zemris.nnao.neuralNetwork.activations.IActivation;

import java.io.IOException;
import java.util.List;

public class BPTestLinear {
    // PROOF OF CONCEPT!
    //OVO DODE DO mse od 98 na training setu
    public static void main(String[] args) throws IOException {
        NeuralNetwork nn = new NeuralNetwork(
                new int[]{1, 6, 1},
                new IActivation[]{ActivationFunctions.Identity, ActivationFunctions.ReLU, ActivationFunctions.ReLU}
        );

        List<DatasetEntry> dataset = DatasetUtils.createLinear();


        Backpropagation bp = new Backpropagation(dataset.subList(0, 95), dataset.subList(95, dataset.size()), 1E-6,
                100_000, 0., 0.1, nn, 30);
        bp.run();
    }
}
