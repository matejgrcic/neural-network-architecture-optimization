package hr.fer.zemris.nnao.bp;

import hr.fer.zemris.nnao.datasets.DatasetEntry;
import hr.fer.zemris.nnao.datasets.DatasetUtils;
import hr.fer.zemris.nnao.neuralNetwork.NeuralNetwork;
import hr.fer.zemris.nnao.neuralNetwork.activations.ActivationFunctions;
import hr.fer.zemris.nnao.neuralNetwork.activations.IActivation;

import java.io.IOException;
import java.util.List;

public class BPTestSinX {
    // PROOF OF CONCEPT!
    //OVO DODE DO mse od 98 na training setu
    public static void main(String[] args) throws IOException {
        NeuralNetwork nn = new NeuralNetwork(
                new int[]{1, 6, 6, 1},
                new IActivation[]{ActivationFunctions.Identity, ActivationFunctions.Sigmoid, ActivationFunctions.Sigmoid, ActivationFunctions.Identity}
        );

        List<DatasetEntry> dataset = DatasetUtils.createSinXDatasetNormalized();


        Backpropagation bp = new Backpropagation(dataset.subList(0, 110), dataset.subList(110, dataset.size()), 1E-7,
                100_000, 0., 0.1, nn, 30);
        bp.run();
    }
}
