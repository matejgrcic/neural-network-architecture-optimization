package hr.fer.zemris.nnao.datasets;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;
import java.util.function.Function;

public class DatasetUtils {

    public static List<DatasetEntry<Integer, Integer>> createDataset(Path path, String delimiter,
                                                                     Function<String[], DatasetEntry<Integer, Integer>> recordCreator) throws IOException {
        Scanner sc = new Scanner(Files.newInputStream(path));
        List<DatasetEntry<Integer, Integer>> dataset = new ArrayList<>();
        while (sc.hasNextLine()) {
            String[] lineData = sc.nextLine().split(delimiter);
            DatasetEntry<Integer, Integer> record = recordCreator.apply(lineData);
            dataset.add(record);
        }

        return dataset;
    }
}
