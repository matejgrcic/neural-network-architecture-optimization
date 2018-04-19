package test;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;

public class DatasetGeneratorLinear {

    public static void main(String[] args) throws IOException {

        double minValue = 0;
        double maxValue = 1;
        double delta = 0.01;

        try (OutputStream os = Files.newOutputStream(Paths.get("./linear.csv"), StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)) {
            for (double i = minValue; i <= maxValue; i += delta) {
                String result = String.format("%.4f,%.4f\n", i, i);
                os.write(result.getBytes());


            }
        }
        System.out.println("Gotov");

    }
}
