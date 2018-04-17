package hr.fer.zemris.nnao.geneticAlgorithms;

import hr.fer.zemris.nnao.neuralNetwork.activations.IActivation;

import java.util.Arrays;
import java.util.Objects;

public class Solution {

    private IActivation[] activations;
    private int numberOfLayers;
    private int[] architecture;
    private double[] weights;
    private double fitness;

    public Solution(IActivation[] activations, int numberOfLayers, int[] architecture, double[] weights) {
        this.activations = activations;
        this.numberOfLayers = numberOfLayers;
        this.architecture = architecture;
        this.weights = weights;
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

    public int[] getArchitecture() {
        return architecture;
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

//    public String serializeSolution() {
//        StringBuilder sb = new StringBuilder();
//        sb.append(numberOfLayers+",");
//        for(int layer: architecture) {
//            sb.append(layer + ",");
//        }
//        for(IActivation activation: activations) {
//            sb.append(activation.getStringRepresentation() + ",");
//        }
//        String result = sb.toString();
//        return result.substring(0,result.length()-1);
//    }
}
