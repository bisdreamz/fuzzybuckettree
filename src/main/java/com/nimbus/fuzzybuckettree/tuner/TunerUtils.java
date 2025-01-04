package com.nimbus.fuzzybuckettree.tuner;

import java.util.*;

public class TunerUtils {

    public static <T> List<List<T>> generatePermutations(List<T> input) {
        List<List<T>> result = new ArrayList<>();
        if (input.size() == 0) {
            result.add(new ArrayList<>());
            return result;
        }

        T firstElement = input.get(0);
        List<T> restOfList = new ArrayList<>(input.subList(1, input.size()));

        List<List<T>> subPermutations = generatePermutations(restOfList);

        for (List<T> subPermutation : subPermutations) {
            for (int i = 0; i <= subPermutation.size(); i++) {
                List<T> newPermutation = new ArrayList<>(subPermutation);
                newPermutation.add(i, firstElement);
                result.add(newPermutation);
            }
        }

        return result;
    }

    public static <T> List<List<T>> generateRandomPermutations(List<T> input) {
        List<List<T>> result = new ArrayList<>();
        Random random = new Random();

        // First add original list as one permutation
        result.add(new ArrayList<>(input));

        // Generate rest of permutations using Fisher-Yates shuffle
        for (int p = 1; p < factorial(input.size()); p++) {
            List<T> newPerm = new ArrayList<>(input);
            for (int i = newPerm.size() - 1; i > 0; i--) {
                int j = random.nextInt(i + 1);
                // Swap elements
                T temp = newPerm.get(i);
                newPerm.set(i, newPerm.get(j));
                newPerm.set(j, temp);
            }
            result.add(newPerm);
        }

        // Shuffle the final list of permutations too
        Collections.shuffle(result);
        return result;
    }

    private static long factorial(int n) {
        if (n <= 1) return 1;
        return n * factorial(n - 1);
    }

    public static List<Map<String, float[]>> generateBucketCombinations(List<FeatureBucketOptions> features) {
        List<Map<String, float[]>> results = new ArrayList<>();

        // Start with the first feature's combinations
        FeatureBucketOptions firstFeature = features.get(0);
        if (firstFeature.bucketOptions() == null || firstFeature.bucketOptions().length == 0) {
            // For features without buckets, use exact matching
            Map<String, float[]> combo = new HashMap<>();
            combo.put(firstFeature.label(), new float[0]);
            results.add(combo);
        } else {
            float[] bucketOptions = firstFeature.bucketOptions();
            int numValues = firstFeature.valuesCount();

            if (numValues == 1) {
                // Single value feature
                for (float bucketValue : bucketOptions) {
                    Map<String, float[]> combo = new HashMap<>();
                    combo.put(firstFeature.label(), new float[]{ bucketValue });
                    results.add(combo);
                }
            } else {
                // Multi-value feature - generate initial combinations
                generateMultiValueCombinations(firstFeature.label(), bucketOptions, numValues, new HashMap<>(), results);
            }
        }

        // Process remaining features
        for (int i = 1; i < features.size(); i++) {
            List<Map<String, float[]>> newResults = new ArrayList<>();
            FeatureBucketOptions feature = features.get(i);

            // For each existing partial combination
            for (Map<String, float[]> partial : results) {
                if (feature.bucketOptions() == null || feature.bucketOptions().length == 0) {
                    // For features without buckets, use exact matching
                    Map<String, float[]> newCombo = new HashMap<>(partial);
                    newCombo.put(feature.label(), new float[0]);
                    newResults.add(newCombo);
                } else {
                    float[] bucketOptions = feature.bucketOptions();
                    int numValues = feature.valuesCount();

                    if (numValues == 1) {
                        // Single value features
                        for (float bucketValue : bucketOptions) {
                            Map<String, float[]> newCombo = new HashMap<>(partial);
                            newCombo.put(feature.label(), new float[]{ bucketValue });
                            newResults.add(newCombo);
                        }
                    } else {
                        // Multi-value features
                        generateMultiValueCombinations(feature.label(), bucketOptions, numValues, partial, newResults);
                    }
                }
            }
            results = newResults;
        }

        return results;
    }

    private static void generateMultiValueCombinations(String label, float[] bucketOptions, int numValues,
                                                       Map<String, float[]> partial,
                                                       List<Map<String, float[]>> results) {
        int[] counters = new int[numValues];
        boolean hasMore = true;

        while (hasMore) {
            Map<String, float[]> newCombo = new HashMap<>(partial);
            float[] buckets = new float[numValues];

            for (int i = 0; i < numValues; i++) {
                buckets[i] = bucketOptions[counters[i]];
            }
            newCombo.put(label, buckets);
            results.add(newCombo);

            hasMore = false;
            for (int i = numValues - 1; i >= 0; i--) {
                counters[i]++;
                if (counters[i] >= bucketOptions.length) {
                    counters[i] = 0;
                } else {
                    hasMore = true;
                    break;
                }
            }
        }
    }

}
