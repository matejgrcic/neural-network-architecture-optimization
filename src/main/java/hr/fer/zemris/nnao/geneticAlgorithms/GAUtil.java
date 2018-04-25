package hr.fer.zemris.nnao.geneticAlgorithms;

import java.util.Comparator;

public class GAUtil {

    public static Comparator<Solution> createSolutionComparator(double solutionDelta) {
        return (s1, s2) -> Double.compare(s1.getFitness(), s2.getFitness());
    }
}
