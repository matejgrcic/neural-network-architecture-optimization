package test;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;

public class DatasetGenerator {

    public static void main(String[] args) throws IOException {

        double minValue = -5.12;
        double maxValue = 5.12;
        double delta = 0.64;

        try (OutputStream os = Files.newOutputStream(Paths.get("./rastring2D.csv"), StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)) {
            for (double i = minValue; i <= maxValue; i += delta) {
                for (double j = minValue; j <= maxValue; j += delta) {
                        double value = 10.* 2.
                                + (i*i - 10. * Math.cos(2.*Math.PI*i))
                                + (j*j - 10. * Math.cos(2.*Math.PI*j));
                        String result = String.format("%.4f,%.4f,%.4f\n",i,j,value);
                        os.write(result.getBytes());

                }
            }
        }
        System.out.println("Gotov");

    }
}
