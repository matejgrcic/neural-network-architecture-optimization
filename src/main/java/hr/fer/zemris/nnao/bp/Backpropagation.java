package hr.fer.zemris.nnao.bp;

import hr.fer.zemris.nnao.datasets.DatasetEntry;
import hr.fer.zemris.nnao.geneticAlgorithms.Solution;
import hr.fer.zemris.nnao.neuralNetwork.INeuralNetwork;
import hr.fer.zemris.nnao.neuralNetwork.NNUtil;
import hr.fer.zemris.nnao.neuralNetwork.NeuralNetwork;
import hr.fer.zemris.nnao.neuralNetwork.activations.IActivation;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import java.util.List;

import static hr.fer.zemris.nnao.bp.BPUtil.createBatches;
import static hr.fer.zemris.nnao.neuralNetwork.NNUtil.*;

public class Backpropagation extends AbstractBackpropagation {

    public Backpropagation(List<DatasetEntry> trainingSet, List<DatasetEntry> validationSet,
                           double learningRate, long maxIteration, double desiredError,
                           double desiredPrecision, NeuralNetwork neuralNetwork, int batchSize) {

        super(trainingSet, validationSet, learningRate, maxIteration,
                desiredError, desiredPrecision, neuralNetwork, batchSize);
    }

    //vraca validation set mse
    public double run() {
        List<DatasetEntry>[] batches = createBatches(batchSize, trainingSet);
        int[] nnArchitecture = neuralNetwork.getNeuralNetworkArchitecture();

        while (currentIteration <= maxIteration) {
            RealVector totalErrorByNeuron = new ArrayRealVector(nnArchitecture[nnArchitecture.length - 1]);
            for (List<DatasetEntry> batch : batches) {
                RealMatrix outputDeltaMatrix = new Array2DRowRealMatrix(batch.size(), nnArchitecture[nnArchitecture.length - 1]);
                RealMatrix[] layerOutputsByBatch = new RealMatrix[nnArchitecture.length];
                for (int i = 0; i < nnArchitecture.length - 1; ++i) {
                    layerOutputsByBatch[i] = new Array2DRowRealMatrix(batch.size(), nnArchitecture[i] + 1);
                }
                layerOutputsByBatch[layerOutputsByBatch.length - 1] = new Array2DRowRealMatrix(batch.size(), nnArchitecture[nnArchitecture.length - 1]);
                for (int i = 0; i < batch.size(); ++i) {
                    DatasetEntry entry = batch.get(i);
                    double[] output = neuralNetwork.forward(entry.getInput());
                    RealVector calculatedOutput = new ArrayRealVector(output);
                    RealVector expectedOutput = new ArrayRealVector(entry.getOutput());
                    RealVector error = expectedOutput.subtract(calculatedOutput);
                    outputDeltaMatrix.setRowVector(i, error);
                    totalErrorByNeuron = totalErrorByNeuron.add(error.ebeMultiply(error));

                    double[][] outputOfLayers = neuralNetwork.getOutputOfLayers();
                    for (int j = 0; j < outputOfLayers.length; ++j) {
                        layerOutputsByBatch[j].setRow(i, outputOfLayers[j]);
                    }
                }
                doBackpropagation(neuralNetwork, outputDeltaMatrix, layerOutputsByBatch);
            }
            //u metodu
            double mseSum = 0;
            for (double totalNeuronError : totalErrorByNeuron.toArray()) {
                mseSum += totalNeuronError;
            }
            trainingMSE = mseSum / trainingSet.size();
//            System.out.println("Iter : " + currentIteration + " MSError je: " + trainingMSE);

            RealVector validationMse = new ArrayRealVector(nnArchitecture[nnArchitecture.length - 1]);
            for (DatasetEntry entry : validationSet) {
                RealVector output = new ArrayRealVector(neuralNetwork.forward(entry.getInput()));
                RealVector expectedOutput = new ArrayRealVector(entry.getOutput());
                RealVector error = output.subtract(expectedOutput);
                validationMse = validationMse.add(error.ebeMultiply(error));
            }

            //u metodu
            double validationMseSum = 0;
            for (double totalNeuronError : totalErrorByNeuron.toArray()) {
                validationMseSum += totalNeuronError;
            }
            double lastIterValMSE = validationMSE;
            validationMSE = validationMseSum / validationSet.size();
//            System.out.println("Iter : " + currentIteration + " Validation MSError je: " + validationMSE);
            if (lastIterValMSE <= validationMSE && currentIteration > maxIteration / 2) {
                break;
            }

            if(Double.isNaN(trainingMSE)){
                break;
            }

            if (Math.abs(trainingMSE - desiredError) < desiredPrecision) {
                break;
            }

            ++currentIteration;
        }
//        System.out.println("MSError je: " + trainingMSE);
        return trainingMSE;
    }

    private double[] doBackpropagation(INeuralNetwork neuralNetwork, RealMatrix outputDeltaMatrix, RealMatrix[] allLayerOutputs) {
        RealMatrix outputLayerError = new Array2DRowRealMatrix(
                outputDeltaMatrix.getRowDimension(), outputDeltaMatrix.getColumnDimension()
        );
        for (int i = 0; i < outputDeltaMatrix.getRowDimension(); ++i) {
            RealVector outputDerived = allLayerOutputs[allLayerOutputs.length - 1].getRowVector(i)
                    .map(t -> neuralNetwork.getActivationFunctions()[neuralNetwork.getActivationFunctions().length - 1].calculateFirstDerivative(t)
                    );
            outputLayerError.setRowVector(i, outputDerived.ebeMultiply(outputDeltaMatrix.getRowVector(i)));
        }
        RealMatrix[] layerWeights = neuralNetwork.getWeightsMatrix();
        RealMatrix lastLayerWeights = layerWeights[neuralNetwork.getActivationFunctions().length - 2];
        RealMatrix lastHiddenLayerOutput = allLayerOutputs[allLayerOutputs.length - 2];
        for (int i = 0; i < lastLayerWeights.getRowDimension(); ++i) {
            RealVector yOutput = lastHiddenLayerOutput.getColumnVector(i);
            for (int j = 0; j < lastLayerWeights.getColumnDimension(); ++j) {
                double sum = outputLayerError.getColumnVector(j).dotProduct(yOutput);
                double currentWeight = lastLayerWeights.getEntry(i, j);
                lastLayerWeights.setEntry(i, j, currentWeight + sum * learningRate);
            }
        }

        RealMatrix currentError = outputLayerError;
        IActivation[] activations = neuralNetwork.getActivationFunctions();
        for (int i = layerWeights.length - 1; i > 0; --i) {
            RealMatrix nextLayerMatrix = layerWeights[i];
            RealMatrix currentLayerOutput = allLayerOutputs[i];
            RealMatrix currentLayerError = new Array2DRowRealMatrix(
                    currentLayerOutput.getRowDimension(), currentLayerOutput.getColumnDimension()
            );
            final int index = i;
            for (int j = 0; j < currentLayerOutput.getRowDimension(); ++j) {
                RealVector row = currentLayerOutput.getRowVector(j);
                row = row.map(t -> activations[index].calculateFirstDerivative(t));
                for (int k = 0; k < row.getDimension(); ++k) {
                    double err = nextLayerMatrix.getRowVector(k).dotProduct(currentError.getRowVector(j));
                    currentLayerError.setEntry(j, k, err * row.getEntry(k));
                }
            }
            currentError = currentLayerError.getSubMatrix(0, currentLayerError.getRowDimension() - 1,
                    0, currentLayerError.getColumnDimension() - 2);

            RealMatrix currentWeights = layerWeights[i - 1];
            RealMatrix currentOutput = allLayerOutputs[i - 1];
            for (int j = 0; j < currentWeights.getRowDimension(); j++) {
                for (int k = 0; k < currentWeights.getColumnDimension(); ++k) {
                    double currentWeight = currentWeights.getEntry(j, k);
                    double gradient = currentError.getColumnVector(k).dotProduct(currentOutput.getColumnVector(j));
                    currentWeights.setEntry(j, k, currentWeight + learningRate * gradient);
                }
            }
        }
        double[] weights = getWeights(calculateNumberOfWeights(neuralNetwork.getNeuralNetworkArchitecture()),
                neuralNetwork.getWeightsMatrix());
        return weights;
    }
}
