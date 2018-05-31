package hr.fer.zemris.nnao.geneticAlgorithms;

import hr.fer.zemris.nnao.neuralNetwork.activations.IActivation;

import java.util.Arrays;
import java.util.Objects;

public class Solution {

    private IActivation[] activations;
    private int[] layers;
    private int numberOfLayers;

    private double[] weights;
    private double fitness;

    public Solution(IActivation[] activations, int[] layers, int numberOfLayers) {
        this.activations = activations;
        this.layers = layers;
        this.numberOfLayers = numberOfLayers;
    }

    public void setWeights(double[] weights) {
        this.weights = weights;
    }

    public IActivation[] getActivations() {
        return activations;
    }

    public int getNumberOfLayers() {
        return numberOfLayers;
    }

    public int[] getLayers() {
        return layers;
    }

    public void setArchitecture(int[] layers, IActivation[] activations) {
        this.layers = layers;
        this.activations = activations;
    }

    public void setLayer(int layer, int index) {
        layers[index] = layer;
    }

    public  void setActivation(IActivation activation, int index) {
        activations[index] = activation;
    }

    public void setNumberOfLayers(int numberOfLayers) {
        this.numberOfLayers = numberOfLayers;
    }

    public double[] getWeights() {
        return weights;
    }

    public double getFitness() {
        return fitness;
    }

    public void setFitness(double fitness) {
        this.fitness = fitness;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder(numberOfLayers+": ");
        for(int i = 0; i<numberOfLayers; ++i) {
            sb.append("["+layers[i]+ "/"+activations[i]+"] / ");
        }
        return sb.toString();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        Solution solution = (Solution) o;

        if (numberOfLayers != solution.numberOfLayers) return false;
        if (!Arrays.equals(activations, solution.activations)) return false;
        if (!Arrays.equals(layers, solution.layers)) return false;
        return Arrays.equals(weights, solution.weights);
    }

    @Override
    public int hashCode() {
        int result = Arrays.hashCode(activations);
        result = 31 * result + numberOfLayers;
        result = 31 * result + Arrays.hashCode(layers);
        return result;
    }
}
