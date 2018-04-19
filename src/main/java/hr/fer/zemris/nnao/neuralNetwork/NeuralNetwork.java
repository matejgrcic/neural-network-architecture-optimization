package hr.fer.zemris.nnao.neuralNetwork;

import hr.fer.zemris.nnao.neuralNetwork.activations.IActivation;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import static hr.fer.zemris.nnao.neuralNetwork.NNUtil.*;

public class NeuralNetwork implements INeuralNetwork {

    private RealMatrix[] weights;
    private IActivation[] activationFunctions;
    private int[] architecture;
    private int weightsNumber;
    private double[][] outputOfLayers;

    public NeuralNetwork(int[] architecture, IActivation[] activationFunctions) {
        weights = createWeightMatrices(architecture);
        for (RealMatrix matrix : weights) {
            weightsNumber += matrix.getColumnDimension() * matrix.getRowDimension();
        }
        this.activationFunctions = activationFunctions;
        this.architecture = architecture;
    }

    // vraca output
    public double[] forward(double[] input) {
        outputOfLayers = new double[architecture.length][];
        RealVector inputVector = new ArrayRealVector(input);
        inputVector = inputVector.map(t -> activationFunctions[0].calculateValue(t));
        inputVector = inputVector.append(1.);
        outputOfLayers[0] = inputVector.toArray();
        for (int i = 0; i < weights.length-1; ++i) {
            inputVector = weights[i].preMultiply(inputVector);
            final int k = i;
            inputVector = inputVector.map(t -> activationFunctions[k + 1].calculateValue(t));
            inputVector = inputVector.append(1.);
            outputOfLayers[i + 1] = inputVector.toArray();
        }

        inputVector = weights[weights.length-1].preMultiply(inputVector);
        inputVector = inputVector.map(t -> activationFunctions[activationFunctions.length-1].calculateValue(t));
        outputOfLayers[outputOfLayers.length-1] = inputVector.toArray();
        return outputOfLayers[outputOfLayers.length-1];
    }


    public IActivation[] getActivationFunctions() {
        return activationFunctions;
    }

    public void setActivationFunctions(IActivation[] activationFunctions) {
        this.activationFunctions = activationFunctions;
    }

    public int[] getNeuralNetworkArchitecture() {
        return architecture;
    }

    public RealMatrix[] getWeightsMatrix() {
        return weights;
    }

    public int getWeightsNumber() {
        return weightsNumber;
    }

    public void setWeights(double[] weightsArray) {
        int offset = 0;
        for (RealMatrix matrix : weights) {
            for (int i = 0; i < matrix.getRowDimension(); ++i) {
                for (int j = 0; j < matrix.getColumnDimension(); ++j) {
                    matrix.setEntry(i, j, weightsArray[offset++]);
                }
            }
        }
    }

    public void setWeights(RealMatrix[] weights) {
        this.weights = weights;
    }


    public int getOutputSize() {
        return architecture[architecture.length - 1];
    }

    public double[][] getOutputOfLayers() {
        return outputOfLayers;
    }
}
