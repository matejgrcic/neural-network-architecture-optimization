package hr.fer.zemris.nnao.swarmAlgorithms;

import hr.fer.zemris.nnao.datasets.DatasetEntry;
import hr.fer.zemris.nnao.datasets.DatasetUtils;
import hr.fer.zemris.nnao.neuralNetwork.NeuralNetwork;
import hr.fer.zemris.nnao.neuralNetwork.activations.ActivationFunctions;
import hr.fer.zemris.nnao.neuralNetwork.activations.IActivation;

import java.io.IOException;
import java.util.List;
import java.util.function.BiFunction;

public class PSONNExampleIris {
    // PROOF OF CONCEPT!
    //OVO DODE DO mse od 1.666
    public static void main(String[] args) throws IOException {
        NeuralNetwork nn =
                new NeuralNetwork(
                        new int[]{4, 87, 1},
                        new IActivation[]{ActivationFunctions.Identity, ActivationFunctions.ReLU, ActivationFunctions.Identity});
        double[] lowerBound = new double[nn.getWeightsNumber()];
        for (int i = 0; i < lowerBound.length; ++i) {
            lowerBound[i] = -5.12;
        }
        double[] upperBound = new double[nn.getWeightsNumber()];
        for (int i = 0; i < lowerBound.length; ++i) {
            upperBound[i] = 5.12;
        }
        double[] lowerSpeed = new double[nn.getWeightsNumber()];
        for (int i = 0; i < lowerBound.length; ++i) {
            lowerSpeed[i] = -2.;
        }
        double[] upperSpeed = new double[nn.getWeightsNumber()];
        for (int i = 0; i < lowerBound.length; ++i) {
            upperSpeed[i] = 2.;
        }

        List<DatasetEntry> data = DatasetUtils.createIrisDataset();

        BiFunction<Double, Double, Boolean> comparator = (t, u) -> Math.abs(t) > Math.abs(u);

        AlgorithmPSO pso = new AlgorithmPSO(50, nn.getWeightsNumber(), lowerBound, upperBound, lowerSpeed, upperSpeed);
        double[] result = pso.run(t -> {
            nn.setWeights(t);
            double sum = 0.;
            for (DatasetEntry d : data) {
                double[] res = nn.forward(d.getInput());
                sum += Math.pow(d.getOutput()[0] - res[0], 2.);
            }
            return sum / data.size();
        }, comparator, 0., 1E-3, 200);

        nn.setWeights(result);
        double sum = 0.;
        for (DatasetEntry d : data) {
            sum += Math.pow(nn.forward(d.getInput())[0] - d.getOutput()[0], 2.);
        }
        System.out.println(sum / data.size());

        int cnt = 0;
        for (DatasetEntry d : data) {
            double[] res = nn.forward(d.getInput());
            double x = Math.round(res[0]);

            double target = d.getOutput()[0];
            if (target != x) {
                cnt++;
            }


        }
        System.out.println("Falilo: " + cnt);
    }
}
