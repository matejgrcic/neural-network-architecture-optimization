package hr.fer.zemris.nnao.neuralNetwork;

import hr.fer.zemris.nnao.neuralNetwork.activations.IActivation;
import org.apache.commons.math3.linear.RealMatrix;

public interface INeuralNetwork {

    double[] forward(double[] input);
    IActivation[] getActivationFunctions();
    int[] getNeuralNetworkArchitecture();
    RealMatrix[] getWeightsMatrix();
    void setWeights(double[] weightsArray);
    void setWeights(RealMatrix[] weights);
}
