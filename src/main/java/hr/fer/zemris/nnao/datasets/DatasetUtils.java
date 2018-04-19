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

    public static List<DatasetEntry> createDataset(Path path, String delimiter,
                                                   Function<String[], DatasetEntry> recordCreator) throws IOException {
        Scanner sc = new Scanner(Files.newInputStream(path));
        List<DatasetEntry> dataset = new ArrayList<>();
        while (sc.hasNextLine()) {
            String[] lineData = sc.nextLine().split(delimiter);
            DatasetEntry record = recordCreator.apply(lineData);
            dataset.add(record);
        }

        return dataset;
    }

    public static List<DatasetEntry> createRastring2DDataset() throws IOException {
        Function<String[], DatasetEntry> creator = u -> {
            double[] input = new double[]{Double.parseDouble(u[0]), Double.parseDouble(u[1])};
            double[] output = new double[]{Double.parseDouble(u[2])};
            return new DatasetEntry(input, output);
        };
        return createDataset(Paths.get("./rastring2D.csv"), ",", creator);
    }

    public static List<DatasetEntry> createSinXDataset() throws IOException {
        Function<String[], DatasetEntry> creator = u -> {
            double[] input = new double[]{Double.parseDouble(u[0])};
            double[] output = new double[]{Double.parseDouble(u[1])};
            return new DatasetEntry(input, output);
        };
        return createDataset(Paths.get("./sinx.csv"), ",", creator);
    }

    public static List<DatasetEntry> createSinXDatasetNormalized() throws IOException {
        Function<String[], DatasetEntry> creator = u -> {
            double[] input = new double[]{Double.parseDouble(u[0])};
            double[] output = new double[]{Double.parseDouble(u[1])};
            return new DatasetEntry(input, output);
        };
        return createDataset(Paths.get("./sinxN01.csv"), ",", creator);
    }

    public static List<DatasetEntry> createLinear() throws IOException {
        Function<String[], DatasetEntry> creator = u -> {
            double[] input = new double[]{Double.parseDouble(u[0])};
            double[] output = new double[]{Double.parseDouble(u[1])};
            return new DatasetEntry(input, output);
        };
        return createDataset(Paths.get("./linear.csv"), ",", creator);
    }
}
