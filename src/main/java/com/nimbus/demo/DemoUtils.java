package com.nimbus.demo;

import com.nimbus.fuzzybuckettree.tuner.TrainingEntry;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class DemoUtils {

    public static List<TrainingEntry<String>> loadCarData() {
        List<TrainingEntry<String>> data = new ArrayList<>();

        try {
            InputStream inputStream = DemoUtils.class.getResourceAsStream("/cars.csv");
            if (inputStream == null) {
                throw new RuntimeException("Cannot find cars.csv in resources folder");
            }

            BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream));
            String line;

            while ((line = reader.readLine()) != null) {
                Map<String, String[]> features = new HashMap<>();

                String[] values = line.trim().split(",");

                int idx = 0;
                for (String val : values) {
                    if (++idx == values.length) {
                        data.add(new TrainingEntry<>(features, val.trim()));
                        break;
                    }

                    String label = switch (idx-1) {
                        case 0 -> "price";
                        case 1 -> "maint";
                        case 2 -> "doors";
                        case 3 -> "persons";
                        case 4 -> "lugboot";
                        case 5 -> "safety";
                        default -> throw new RuntimeException("Encountered unknown car data sample index");
                    };

                    features.put(label, new String[]{ val });
                }
            }

            reader.close();
        } catch (Exception e) {
            throw new RuntimeException("Error loading car data: " + e.getMessage(), e);
        }

        return data;
    }

}
