package hr.fer.zemris.nnao.swarmAlgorithms;

import java.util.function.BiFunction;
import java.util.function.Function;

public class PSOExample {

    public static void main(String[] args) {

        AlgorithmPSO pso = new AlgorithmPSO(50, 2, new double[]{-5.12, -5.12}, new double[]{5.12, 5.12},
                new double[]{-2., -2.}, new double[]{2., 2.});

        Function<double[], Double> evaluator = u -> {
            double sum = 0.;
            for (double t : u) {
                sum += t * t - 10. * Math.cos(2. * Math.PI * t);
            }
            return sum + 10. * u.length;
        };

        BiFunction<Double, Double, Boolean> comparator = (t, u) -> Math.abs(t) > Math.abs(u);

        double[] result = pso.run(evaluator, comparator, 0., 1E-3, 100);
//        System.out.println("Best solution x: " + String.format("%.3f", result[0]) + " y: " + String.format("%.3f", result[1]));
    }
}
