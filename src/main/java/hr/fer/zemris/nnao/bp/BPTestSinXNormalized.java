package hr.fer.zemris.nnao.bp;

import hr.fer.zemris.nnao.datasets.DatasetEntry;
import hr.fer.zemris.nnao.datasets.DatasetUtils;
import hr.fer.zemris.nnao.neuralNetwork.INeuralNetwork;
import hr.fer.zemris.nnao.neuralNetwork.NeuralNetwork;
import hr.fer.zemris.nnao.neuralNetwork.activations.ActivationFunctions;
import hr.fer.zemris.nnao.neuralNetwork.activations.IActivation;

import java.io.IOException;
import java.util.List;

public class BPTestSinXNormalized {
    // PROOF OF CONCEPT!
    //OVO DODE DO 0.1 !!!!!
    public static void main(String[] args) throws IOException {
        INeuralNetwork nn = new NeuralNetwork(
                new int[]{1, 10, 5, 1},
                new IActivation[]{ActivationFunctions.Identity, ActivationFunctions.ReLU, ActivationFunctions.Sigmoid, ActivationFunctions.Identity}
        );

        List<DatasetEntry> dataset = DatasetUtils.createSinXDatasetNormalized();


        Backpropagation bp = new Backpropagation(dataset.subList(0, 115), dataset.subList(115, dataset.size()), 1E-4,
                50_000, 0., 0.1, nn, 30);
        bp.run();
    }
}
