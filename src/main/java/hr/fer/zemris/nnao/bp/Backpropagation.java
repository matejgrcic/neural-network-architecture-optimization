package hr.fer.zemris.nnao.bp;

import hr.fer.zemris.nnao.datasets.DatasetEntry;
import hr.fer.zemris.nnao.neuralNetwork.INeuralNetwork;
import hr.fer.zemris.nnao.neuralNetwork.NeuralNetwork;
import hr.fer.zemris.nnao.neuralNetwork.activations.IActivation;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import java.util.List;

import static hr.fer.zemris.nnao.bp.BPUtil.createBatches;
import static hr.fer.zemris.nnao.neuralNetwork.NNUtil.calculateNumberOfWeights;
import static hr.fer.zemris.nnao.neuralNetwork.NNUtil.getWeights;

public class Backpropagation extends AbstractBackpropagation {

    public Backpropagation(List<DatasetEntry> trainingSet, List<DatasetEntry> validationSet,
                           double learningRate, long maxIteration, double desiredError,
                           double desiredPrecision, INeuralNetwork neuralNetwork, int batchSize) {

        super(trainingSet, validationSet, learningRate, maxIteration,
                desiredError, desiredPrecision, neuralNetwork, batchSize);
    }

    public double run() {
        List<DatasetEntry>[] batches = createBatches(batchSize, trainingSet);
        int[] nnArchitecture = neuralNetwork.getNeuralNetworkArchitecture();

        while (currentIteration <= maxIteration) {
            RealVector totalErrorByNeuron = null;
            for (List<DatasetEntry> batch : batches) {
                RealMatrix outputDeltaMatrix = new Array2DRowRealMatrix(batch.size(), nnArchitecture[nnArchitecture.length - 1]);
                RealMatrix[] layerOutputsByBatch = createOutputMatrices(nnArchitecture, batch);
                totalErrorByNeuron = fillOutputMatrices(outputDeltaMatrix, layerOutputsByBatch, nnArchitecture[nnArchitecture.length - 1], batch);

                double[] weights = doBackpropagation(neuralNetwork, outputDeltaMatrix, layerOutputsByBatch);
                neuralNetwork.setWeights(weights);
            }
            trainingMSE = calculateMse(totalErrorByNeuron,trainingSet.size());

            RealVector validationMse = fillErrorVector(nnArchitecture[nnArchitecture.length-1],validationSet, neuralNetwork);

            double lastIterValMSE = validationMSE;
            validationMSE = calculateMse(validationMse,validationSet.size());

            if (lastIterValMSE <= validationMSE && currentIteration > maxIteration / 2) {
                validationMSE = lastIterValMSE;
                break;
            }

            if (Double.isNaN(trainingMSE) || Math.abs(trainingMSE - desiredError) < desiredPrecision) {
                break;
            }

            ++currentIteration;
        }
        return validationMSE;
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

    private RealMatrix[] createOutputMatrices(int[] nnArchitecture, List<DatasetEntry> batch) {
        RealMatrix[] outputMatrices = new RealMatrix[nnArchitecture.length];
        for (int i = 0; i < nnArchitecture.length - 1; ++i) {
            outputMatrices[i] = new Array2DRowRealMatrix(batch.size(), nnArchitecture[i] + 1);
        }
        outputMatrices[outputMatrices.length - 1] = new Array2DRowRealMatrix(batch.size(), nnArchitecture[nnArchitecture.length - 1]);

        return outputMatrices;
    }

    private RealVector fillOutputMatrices(RealMatrix outputDeltaMatrix, RealMatrix[] layerOutputsByBatch, int outputSize, List<DatasetEntry> batch) {
        RealVector totalErrorByNeuron = new ArrayRealVector(outputSize);
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
        return totalErrorByNeuron;
    }

    private double calculateMse(RealVector errorVector, int setSize){
        double sum = 0;
        for (double error : errorVector.toArray()) {
            sum += error;
        }
        return sum / setSize;
    }

    private RealVector fillErrorVector(int size, List<DatasetEntry> dataset, INeuralNetwork neuralNetwork) {
        RealVector errorVector = new ArrayRealVector(size);
        for (DatasetEntry entry : dataset) {
            RealVector output = new ArrayRealVector(neuralNetwork.forward(entry.getInput()));
            RealVector expectedOutput = new ArrayRealVector(entry.getOutput());
            RealVector error = output.subtract(expectedOutput);
            errorVector = errorVector.add(error.ebeMultiply(error));
        }
        return errorVector;
    }
}
