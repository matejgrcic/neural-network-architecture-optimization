package hr.fer.zemris.nnao.geneticAlgorithms;

import java.util.Comparator;

public class GAUtil {

    public static Comparator<Solution> createSolutionComparator(double solutionDelta) {
        return (s1, s2) -> {
            if(Math.abs(s1.getFitness() - s2.getFitness())>solutionDelta){
                return (int)(s1.getFitness() - s2.getFitness()*1000);
            }
            return s1.getNumberOfLayers() - s2.getNumberOfLayers();
        };
    }
}
