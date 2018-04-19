package test;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;

public class DatasetGeneratorSinX {

    public static void main(String[] args) throws IOException {

        double minValue = 0;
        double maxValue = 4 * Math.PI;
        double delta = 0.1;

        try (OutputStream os = Files.newOutputStream(Paths.get("./sinxN01.csv"), StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)) {
            for (double i = minValue; i <= maxValue; i += delta) {
                double val = 0.5*(Math.sin(i)+1.);
                String result = String.format("%.4f,%.4f\n", i, val);
                os.write(result.getBytes());


            }
        }
        System.out.println("Gotov");

    }
}
