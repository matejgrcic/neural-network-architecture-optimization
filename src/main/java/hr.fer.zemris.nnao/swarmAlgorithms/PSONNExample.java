package hr.fer.zemris.nnao.swarmAlgorithms;

import hr.fer.zemris.nnao.datasets.DatasetEntry;
import hr.fer.zemris.nnao.datasets.DatasetUtils;
import hr.fer.zemris.nnao.neuralNetwork.activations.ActivationFunctions;
import hr.fer.zemris.nnao.neuralNetwork.activations.IActivation;
import hr.fer.zemris.nnao.neuralNetwork.NeuralNetwork;

import javax.xml.crypto.Data;
import java.io.IOException;
import java.util.List;
import java.util.function.BiFunction;

public class PSONNExample {

    public static void main(String[] args) throws IOException{
        NeuralNetwork nn =
                new NeuralNetwork(
                        new int[]{2, 10, 5, 1},
                        new IActivation[]{ActivationFunctions.Identity, ActivationFunctions.ReLU,ActivationFunctions.ReLU , ActivationFunctions.Identity});
        double[] lowerBound = new double[nn.getWeightsNumber()];
        for (int i = 0; i < lowerBound.length; ++i) {
            lowerBound[i] = -10;
        }
        double[] upperBound = new double[nn.getWeightsNumber()];
        for (int i = 0; i < lowerBound.length; ++i) {
            lowerBound[i] = 10.;
        }
        double[] lowerSpeed = new double[nn.getWeightsNumber()];
        for (int i = 0; i < lowerBound.length; ++i) {
            lowerBound[i] = -1.;
        }
        double[] upperSpeed = new double[nn.getWeightsNumber()];
        for (int i = 0; i < lowerBound.length; ++i) {
            lowerBound[i] = 1.;
        }

        List<DatasetEntry> data = DatasetUtils.createRastring2DDataset();

        BiFunction<Double, Double, Boolean> comparator = (t, u) -> Math.abs(t) > Math.abs(u);


        AlgorithmPSO pso = new AlgorithmPSO(50, nn.getWeightsNumber(), lowerBound, upperBound, lowerSpeed, upperSpeed);
        double[] result = pso.run(t -> {
            nn.setWeights(t);
            double sum = 0.;
            for(DatasetEntry d : data) {
                sum += nn.forward(d.getInput())[0] - d.getOutput()[0];
            }
            return sum;
        }, comparator, 0., 1E-3, 200);
    }
}
