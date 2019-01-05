package hr.fer.zemris.nnao.datasets;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Scanner;
import java.util.function.Function;

public class DatasetUtils {

    public static final String IRIS_DATASET = "./resources/datasets/iris.csv";
    public static final String DELIMITER = ",";

    public static List<DatasetEntry> createDataset(Path path, String delimiter,
                                                   Function<String[], DatasetEntry> recordCreator) throws IOException {
        Scanner sc = new Scanner(Files.newInputStream(path));
        List<DatasetEntry> dataset = new ArrayList<>();
        while (sc.hasNextLine()) {
            String[] lineData = sc.nextLine().split(delimiter);
            DatasetEntry record = recordCreator.apply(lineData);
            dataset.add(record);
        }
        sc.close();
        return dataset;
    }

    public static List<DatasetEntry> createIrisDataset() throws IOException {
        Function<String[], DatasetEntry> creator = u -> {
            double[] input = new double[]{Double.parseDouble(u[0]),Double.parseDouble(u[1]),Double.parseDouble(u[2]),Double.parseDouble(u[3])};
            double[] output = null;
            if(u[4].equals("setosa")) {
                output = new double[]{-1.};
            }else if(u[4].equals("versicolor")) {
                output = new double[]{0.};
            }else if(u[4].equals("virginica")) {
                output = new double[]{1.};
            }else {
                throw new RuntimeException("Invalid dataset value");
            }

            return new DatasetEntry(input, output);
        };
        List<DatasetEntry> dataset = createDataset(Paths.get(IRIS_DATASET), DELIMITER, creator);
        Collections.shuffle(dataset);
        return dataset;
    }
}
