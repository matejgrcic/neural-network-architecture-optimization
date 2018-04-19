package hr.fer.zemris.nnao.swarmAlgorithms;

import hr.fer.zemris.nnao.datasets.DatasetEntry;
import hr.fer.zemris.nnao.datasets.DatasetUtils;
import hr.fer.zemris.nnao.neuralNetwork.NeuralNetwork;
import hr.fer.zemris.nnao.neuralNetwork.activations.ActivationFunctions;
import hr.fer.zemris.nnao.neuralNetwork.activations.IActivation;

import java.io.IOException;
import java.util.List;
import java.util.function.BiFunction;

public class PSONNExampleLinear {
    // PROOF OF CONCEPT!
    //OVO DODE DO mse od 105 na training setu
    public static void main(String[] args) throws IOException{
        NeuralNetwork nn =
                new NeuralNetwork(
                        new int[]{1, 7, 1},
                        new IActivation[]{ActivationFunctions.Identity,ActivationFunctions.Sigmoid , ActivationFunctions.Sigmoid});
        double[] lowerBound = new double[nn.getWeightsNumber()];
        for (int i = 0; i < lowerBound.length; ++i) {
            lowerBound[i] = -5.12;
        }
        double[] upperBound = new double[nn.getWeightsNumber()];
        for (int i = 0; i < lowerBound.length; ++i) {
            upperBound[i] = -5.12;
        }
        double[] lowerSpeed = new double[nn.getWeightsNumber()];
        for (int i = 0; i < lowerBound.length; ++i) {
            lowerSpeed[i] = -0.1;
        }
        double[] upperSpeed = new double[nn.getWeightsNumber()];
        for (int i = 0; i < lowerBound.length; ++i) {
            upperSpeed[i] = 0.1;
        }

        List<DatasetEntry> data = DatasetUtils.createLinear();

        BiFunction<Double, Double, Boolean> comparator = (t, u) -> Math.abs(t) > Math.abs(u);


        AlgorithmPSO pso = new AlgorithmPSO(50, nn.getWeightsNumber(), lowerBound, upperBound, lowerSpeed, upperSpeed);
        double[] result = pso.run(t -> {
            nn.setWeights(t);
            double sum = 0.;
            for(DatasetEntry d : data) {
                sum += Math.pow(nn.forward(d.getInput())[0] - d.getOutput()[0],2.);
            }
            return sum/data.size();
        }, comparator, 0., 1E-3, 100);

        nn.setWeights(result);
        double sum = 0.;
        for(DatasetEntry d : data) {
            sum += Math.pow(nn.forward(d.getInput())[0] - d.getOutput()[0],2.);
        }
        System.out.println(sum/data.size());
    }
}
