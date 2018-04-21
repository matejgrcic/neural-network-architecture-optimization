package hr.fer.zemris.nnao.bp;

import hr.fer.zemris.nnao.datasets.DatasetEntry;
import hr.fer.zemris.nnao.datasets.DatasetUtils;
import hr.fer.zemris.nnao.neuralNetwork.NeuralNetwork;
import hr.fer.zemris.nnao.neuralNetwork.activations.ActivationFunctions;
import hr.fer.zemris.nnao.neuralNetwork.activations.IActivation;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

public class BPTest {
    // PROOF OF CONCEPT!
    //OVO DODE DO mse od 98 na training setu
    public static void main(String[] args ) throws IOException{
        NeuralNetwork nn = new NeuralNetwork(
                new int[] {2, 120,1},
                new IActivation[] {ActivationFunctions.Identity,ActivationFunctions.ReLU, ActivationFunctions.ReLU}
        );

        List<DatasetEntry> dataset = DatasetUtils.createRastring2DDataset();


        Backpropagation bp = new Backpropagation(dataset.subList(0,260),dataset.subList(260,dataset.size()),1E-5,
                100_000, 0., 0.1 , nn, 30);
        bp.run();
    }
}
