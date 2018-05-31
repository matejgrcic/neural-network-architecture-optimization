package hr.fer.zemris.nnao.geneticAlgorithms;

import java.util.Comparator;

public class GAUtil {

    public static Comparator<Solution> solutionComparator = Comparator.comparingDouble(s -> s.getFitness());

    public static int[] integerArrayToIntArrayConverter(Integer[] array) {
        int[] architecture = new int[array.length];
        for(int i = 0; i<array.length; ++i){
            architecture[i] = array[i];
        }
        return architecture;
    }
}
