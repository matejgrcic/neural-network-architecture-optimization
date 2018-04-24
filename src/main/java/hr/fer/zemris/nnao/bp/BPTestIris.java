package hr.fer.zemris.nnao.bp;

import hr.fer.zemris.nnao.datasets.DatasetEntry;
import hr.fer.zemris.nnao.datasets.DatasetUtils;
import hr.fer.zemris.nnao.neuralNetwork.NeuralNetwork;
import hr.fer.zemris.nnao.neuralNetwork.activations.ActivationFunctions;
import hr.fer.zemris.nnao.neuralNetwork.activations.IActivation;

import java.io.IOException;
import java.util.List;

public class BPTestIris {
    // PROOF OF CONCEPT!
    //OVO DODE DO 5 pogresaka!!!!!
    public static void main(String[] args) throws IOException {
        NeuralNetwork nn = new NeuralNetwork(
                new int[]{4, 70, 1},
                new IActivation[]{ActivationFunctions.Identity, ActivationFunctions.Sigmoid, ActivationFunctions.Identity}
        );

        List<DatasetEntry> dataset = DatasetUtils.createIrisDataset();


        Backpropagation bp = new Backpropagation(dataset.subList(0, 105), dataset.subList(105, 128), 1E-4,
                100_000, 0., 1E-3, nn, 30);
        bp.run();

        List<DatasetEntry> test = dataset.subList(128,dataset.size());

        int cnt = 0;
        for (DatasetEntry d : dataset) {
            double[] res = nn.forward(d.getInput());

            double x = Math.round(res[0]);
            double target = d.getOutput()[0];
            System.out.println("Expected: "+target + " Dobiveno: "+ x);
            if(target != x) {
                cnt++;
            }


        }
        System.out.println("Falilo na ukupno: "+cnt);

        cnt = 0;
        for (DatasetEntry d : test) {
            double[] res = nn.forward(d.getInput());

            double x = Math.round(res[0]);
            double target = d.getOutput()[0];
            System.out.println("Expected: "+target + " Dobiveno: "+ x);
            if(target != x) {
                cnt++;
            }


        }
        System.out.println("Falilo na test: "+cnt);

    }
}
