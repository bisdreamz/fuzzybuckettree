package com.nimbus.fuzzybuckettree.tuner;

import java.util.*;

/**
 * A BucketIterator that enumerates only EQUAL, ASCENDING, or DESCENDING patterns.
 * This prevents bucket option patters that oscililate, which can cut down the
 * search space tremendously.
 *
 * - If valuesCount=1, there are exactly #buckets combos (EQUAL only).
 * - If valuesCount>1, we compute union(ASC, DESC):
 *     ascCount + descCount - eqCount
 *   If that exceeds maxCombinations, we attempt "eqCount" only, if <= maxCombinations.
 *   If even eqCount > maxCombinations, we skip entirely.
 */
public class PatternedBucketIterator implements BucketIterator {

    private final List<FeatureBucketOptions> features;
    private final Map<String, Integer> featureSegments;
    private final long totalCombinations;
    private final int batchSize;
    private long currentIndex = 0;

    public enum BucketPattern {
        EQUAL,      // All segments use same bucket
        ASCENDING,  // Each segment must be >= previous
        DESCENDING  // Each segment must be <= previous
    }

    public PatternedBucketIterator(List<FeatureBucketOptions> features, int batchSize) {
        if (features == null || features.isEmpty())
            throw new IllegalArgumentException("Features must not be null or empty");
        if (batchSize <= 0)
            throw new IllegalArgumentException("Batch size must be positive");

        this.features = features;
        this.batchSize = batchSize;
        this.featureSegments = calculateFeatureSegments(features);
        this.totalCombinations = calculateTotalCombinations();
    }

    private Map<String, Integer> calculateFeatureSegments(List<FeatureBucketOptions> features) {
        Map<String, Integer> segments = new HashMap<>();

        for (FeatureBucketOptions feature : features) {
            if (feature.bucketOptions() == null) {
                segments.put(feature.label(), 0);
                continue;
            }

            if (feature.valuesCount() <= 1) {
                segments.put(feature.label(), 1);
                continue;
            }

            // For multi-value features, use feature's maxCombinations
            Integer maxSegments = feature.maxCombinations();
            if (maxSegments == null) {
                throw new IllegalArgumentException("Feature " + feature.label() +
                        " has multiple bucket options but no maxSegments specified");
            }

            // Check if maxSegments is sufficient for bucket options
            if (maxSegments < feature.bucketOptions().length) {
                throw new IllegalArgumentException("Feature " + feature.label() +
                        " maxSegments (" + maxSegments + ") must be >= number of bucket options (" +
                        feature.bucketOptions().length + ") to test all bucket values");
            }

            // Never exceed the feature's value count
            maxSegments = Math.min(maxSegments, feature.valuesCount());
            segments.put(feature.label(), maxSegments);
        }

        return segments;
    }

    private long calculateTotalCombinations() {
        return features.stream()
                .mapToLong(f -> {
                    if (f.bucketOptions() == null) return 1;

                    int segments = featureSegments.get(f.label());
                    if (f.valuesCount() <= 1) {
                        // Single value features only do EQUAL pattern
                        return f.bucketOptions().length;
                    }

                    // Multi-value features try all three patterns
                    return calculatePatternCombinations(segments, f.bucketOptions().length) * 3;
                })
                .reduce(1, (a, b) -> a * b);
    }

    private Map<String, float[]> generateCombination(long index) {
        Map<String, float[]> result = new HashMap<>();
        long remainingIndex = index;

        for (FeatureBucketOptions feature : features) {
            if (feature.bucketOptions() == null) {
                result.put(feature.label(), new float[0]);
                continue;
            }

            int segments = featureSegments.get(feature.label());
            float[] bucketOptions = feature.bucketOptions();

            if (feature.valuesCount() <= 1) {
                // Single value features only do EQUAL pattern
                int bucketIndex = (int)(remainingIndex % bucketOptions.length);
                result.put(feature.label(), new float[]{ bucketOptions[bucketIndex] });
                remainingIndex /= bucketOptions.length;
                continue;
            }

            // For multi-value features, choose pattern and generate values
            int patternIndex = (int)(remainingIndex % 3);
            remainingIndex /= 3;

            BucketPattern pattern = BucketPattern.values()[patternIndex];
            float[] values = generatePatternedSegments(segments, bucketOptions, remainingIndex, pattern);

            // Map segments to full value array
            float[] finalValues = new float[feature.valuesCount()];
            int valuesPerSegment = (feature.valuesCount() + segments - 1) / segments;

            for (int segment = 0; segment < segments; segment++) {
                int startIdx = segment * valuesPerSegment;
                int endIdx = Math.min(startIdx + valuesPerSegment, feature.valuesCount());
                Arrays.fill(finalValues, startIdx, endIdx, values[segment]);
            }

            result.put(feature.label(), finalValues);
            remainingIndex /= calculatePatternCombinations(segments, bucketOptions.length);
        }

        return result;
    }

    // Calculate combinations for a single pattern type
    private long calculatePatternCombinations(int segments, int numBuckets) {
        // For equal pattern: just numBuckets combinations
        // For ascending/descending: formula is (n + k - 1)!/(n!(k-1)!)
        // where n is segments, k is numBuckets
        return factorial(segments + numBuckets - 1) /
                (factorial(segments) * factorial(numBuckets - 1));
    }

    private long factorial(int n) {
        if (n <= 1) return 1;
        long result = 1;
        for (int i = 2; i <= n; i++) {
            result *= i;
            if (result < 0) throw new IllegalArgumentException("Factorial overflow");
        }
        return result;
    }

    @Override
    public List<Map<String, float[]>> getNextBatch() {
        if (!hasMoreBatches()) {
            return Collections.emptyList();
        }

        List<Map<String, float[]>> batch = new ArrayList<>();
        long endIndex = Math.min(currentIndex + batchSize, totalCombinations);
        for (long i = currentIndex; i < endIndex; i++) {
            batch.add(generateCombination(i));
        }

        currentIndex = endIndex;  // Only advance to the actual end
        return batch;
    }

    private float[] generatePatternedSegments(int segments, float[] bucketOptions, long index, BucketPattern pattern) {
        if (bucketOptions == null || bucketOptions.length == 0) {
            throw new IllegalArgumentException("Bucket options cannot be null or empty");
        }
        if (segments <= 0) {
            throw new IllegalArgumentException("Number of segments must be positive");
        }

        float[] result = new float[segments];

        switch (pattern) {
            case EQUAL:
                // Just one bucket value for all segments
                Arrays.fill(result, bucketOptions[(int)(index % bucketOptions.length)]);
                break;

            case ASCENDING: {
                // Start at lowest possible bucket and only increase/stay same
                int bucketIndex = 0;
                for (int i = 0; i < segments; i++) {
                    // Can only use buckets from current index up
                    int availableBuckets = bucketOptions.length - bucketIndex;
                    if (availableBuckets <= 0) {
                        throw new IllegalStateException("No available buckets for ascending pattern");
                    }
                    int nextIncrease = (int)(index % availableBuckets);
                    bucketIndex += nextIncrease;
                    if (bucketIndex >= bucketOptions.length) {
                        throw new IllegalStateException("Bucket index out of bounds: " + bucketIndex);
                    }
                    result[i] = bucketOptions[bucketIndex];
                    index /= availableBuckets;
                }
                break;
            }

            case DESCENDING: {
                // Start at highest possible bucket and only decrease/stay same
                int bucketIndex = bucketOptions.length - 1;
                for (int i = 0; i < segments; i++) {
                    // Can only use buckets from 0 to current index
                    int availableBuckets = bucketIndex + 1;
                    if (availableBuckets <= 0) {
                        throw new IllegalStateException("No available buckets for descending pattern");
                    }
                    int nextDecrease = (int)(index % availableBuckets);
                    bucketIndex -= nextDecrease;
                    if (bucketIndex < 0) {
                        throw new IllegalStateException("Bucket index out of bounds: " + bucketIndex);
                    }
                    result[i] = bucketOptions[bucketIndex];
                    index /= availableBuckets;
                }
                break;
            }

            default:
                throw new IllegalArgumentException("Unsupported pattern: " + pattern);
        }

        return result;
    }

    @Override
    public long getTotalCombinations() {
        return totalCombinations;
    }

    @Override
    public void reset() {
        this.currentIndex = 0;
    }

    @Override
    public int getTotalBatches() {
        return (int)((totalCombinations + batchSize - 1) / batchSize);
    }

    @Override
    public boolean hasMoreBatches() {
        return currentIndex < totalCombinations && currentIndex >= 0;
    }
}