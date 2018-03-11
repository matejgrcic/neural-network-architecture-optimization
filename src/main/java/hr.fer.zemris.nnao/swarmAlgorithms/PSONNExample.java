package hr.fer.zemris.nnao.swarmAlgorithms;

import hr.fer.zemris.nnao.neuralNetwork.activations.ActivationFunctions;
import hr.fer.zemris.nnao.neuralNetwork.activations.IActivation;
import hr.fer.zemris.nnao.neuralNetwork.NeuralNetwork;

import java.util.function.BiFunction;

public class PSONNExample {

    public static void main(String[] args) {
        NeuralNetwork nn =
                new NeuralNetwork(
                        new int[]{5, 3, 1},
                        new IActivation[]{ActivationFunctions.Identity, ActivationFunctions.ReLU, ActivationFunctions.Identity});
        double[] lowerBound = new double[nn.getWeightsNumber()];
        for (int i = 0; i<lowerBound.length; ++i ){
            lowerBound[i] = -5.;
        }
        double[] upperBound = new double[nn.getWeightsNumber()];
        for (int i = 0; i<lowerBound.length; ++i ){
            lowerBound[i] = 5.;
        }
        double[] lowerSpeed = new double[nn.getWeightsNumber()];
        for (int i = 0; i<lowerBound.length; ++i ){
            lowerBound[i] = -2.;
        }
        double[] upperSpeed = new double[nn.getWeightsNumber()];
        for (int i = 0; i<lowerBound.length; ++i ){
            lowerBound[i] = 2.;
        }

        double[] input = new double[] {1.,1.,1.,1.,1.};

        BiFunction<Double, Double, Boolean> comparator = (t, u) -> Math.abs(t) > Math.abs(u);


        AlgorithmPSO pso = new AlgorithmPSO(50, nn.getWeightsNumber(), lowerBound,upperBound,lowerSpeed,upperSpeed);
        double[] result = pso.run(t -> {
            nn.setWeights(t);
            return nn.forward(input)[0] - 5.;
        }, comparator,0., 1E-3, 20);
    }
}
