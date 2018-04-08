//package test;
//
//import hr.fer.zemris.nnao.datasets.DatasetEntry;
//import hr.fer.zemris.nnao.datasets.DatasetUtils;
//import hr.fer.zemris.nnao.neuralNetwork.NeuralNetwork;
//import hr.fer.zemris.nnao.neuralNetwork.activations.ActivationFunctions;
//import hr.fer.zemris.nnao.neuralNetwork.activations.IActivation;
//import hr.fer.zemris.nnao.swarmAlgorithms.AlgorithmPSO;
//
//import java.io.IOException;
//import java.net.Inet4Address;
//import java.nio.file.Files;
//import java.nio.file.Paths;
//import java.util.ArrayList;
//import java.util.List;
//import java.util.Scanner;
//import java.util.function.BiFunction;
//import java.util.function.Function;
//
//public class MnitTest {
//
//    public static void main(String[] args) throws IOException {
//
//        Function<String[], DatasetEntry<Integer, Integer>> dataCreator = t -> {
//            int expected = Integer.parseInt(t[0]);
//            Integer[] input = new Integer[t.length - 1];
//            for (int i = 1; i < t.length; ++i) {
//                input[i - 1] = Integer.parseInt(t[i]);
//            }
//            return new DatasetEntry<>(input, new Integer[]{expected});
//        };
//
//        List<DatasetEntry<Integer, Integer>> dataset = DatasetUtils.createDataset(
//                Paths.get("./datasets/mnist_dataset/mnist_test.csv"),
//                ",",
//                dataCreator);
//
//        List<TestData> inputs = new ArrayList<>(dataset.size());
//        dataset.forEach(t -> {
//            double[] aa = new double[t.getInput().length];
//            int cnt =0;
//            for(int i : t.getInput()) {
//                aa[cnt++] = 1.*i;
//            }
//            inputs.add(new TestData(aa, t.getOutput()[0]));
//            });
//
//        NeuralNetwork nn = new NeuralNetwork(
//                new int[] {784, 50, 10 },
//                new IActivation[] {ActivationFunctions.Identity, ActivationFunctions.ReLU, ActivationFunctions.ReLU,ActivationFunctions.Identity});
//        double[] dmin = new double[nn.getWeightsNumber()];
//        double[] dmax = new double[nn.getWeightsNumber()];
//        double[] vmin = new double[nn.getWeightsNumber()];
//        double[] vmax = new double[nn.getWeightsNumber()];
//        for(int i = 0; i<nn.getWeightsNumber(); ++i) {
//            dmin[i] = -10.12;
//            dmax[i] = 10.12;
//            vmin[i] = -4.;
//            vmax[i] = 4.;
//        }
//
//        AlgorithmPSO pso = new AlgorithmPSO(50, nn.getWeightsNumber(), dmin, dmax,
//                vmin,vmax);
//
//        Function<double[], Double> evaluator = u -> {
//           nn.setWeights(u);
//           int numberOfFails = 0;
//           for(TestData de : inputs) {
//               double[] out = nn.forward(de.input);
//               // softmax
//               double sum = 0.;
//               double best = -Double.MAX_VALUE;
//               int bestIndex = -1;
//               for(int i = 0; i<out.length; ++i) {
//                   double val = Math.exp(out[i]);
//                   sum += val;
//                   if(val > best) {
//                       best = val;
//                       bestIndex = i;
//                   }
//               }
//               if(bestIndex != de.output) {
//                   ++numberOfFails;
//               }
//           }
//           return 1.*numberOfFails;
//        };
//
//        BiFunction<Double, Double, Boolean> comparator = (t, u) -> Math.abs(t) > Math.abs(u);
//
//        double[] result = pso.run(evaluator, comparator, 0., 1E-3, 100);
//        System.out.println("Best solution x: " + String.format("%.3f", result[0]) + " y: " + String.format("%.3f", result[1]));
//
//
//    }
//
//    public static class TestData {
//        double[] input;
//        int output;
//
//        public TestData(double[] input, int output) {
//            this.input = input;
//            this.output = output;
//        }
//
//
//    }
//}
