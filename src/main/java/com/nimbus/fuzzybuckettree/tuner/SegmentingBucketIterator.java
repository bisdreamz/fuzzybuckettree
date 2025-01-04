package com.nimbus.fuzzybuckettree.tuner;

import java.util.*;

/**
 * A bucket combination produces which makes an attempt to explore all
 * possible value/bucket combinations. If there are too many potential
 * combinations, this will divide multi value features into feature
 * segments to reduce the search space. e.g. a bucking pattern of
 * [a,b,c,d] might become [a,a,b,b]. This implementation
 * will search all combinations of bucket options within limits,
 * without enforcing any particular order. [1,3,2,5,3] any
 * combination, including repeating bucket options, will be evaluated.
 */
public class SegmentingBucketIterator implements BucketIterator{

    private final List<FeatureBucketOptions> features;
    private final Map<String, Integer> featureSegments;
    private final long totalCombinations;
    private final int batchSize;
    private long currentIndex;

    /**
     * Construct a segmenting bucket producer which generates all possible bucket options for
     * the given feature value(s). If the combinations exceed the feature's maxCombinations,
     * then multi value features will be segments into the minimum number of contiguous
     * segments required. Otherwise, there are no strict patterns or behaviors applied
     * to the bucking combinations, all will be tested.
     * @param features
     * @param batchSz
     */
    public SegmentingBucketIterator(List<FeatureBucketOptions> features, int batchSz) {
        this.features = features;
        validateFeatures();
        this.featureSegments = calculateFeatureSegments(features);
        this.totalCombinations = calculateTotalCombinations(features, featureSegments);
        this.batchSize = batchSz;
        this.currentIndex = 0;
    }

    private void validateFeatures() {
        for (FeatureBucketOptions feature : features) {
            if (feature.bucketOptions() != null &&
                    feature.bucketOptions().length > 1 &&
                    feature.maxCombinations() == 0 || feature.maxCombinations() > FeatureBucketOptions.MAX_SEGMENTS) {
                throw new IllegalArgumentException(
                        "Feature " + feature.label() + " has multiple bucket options but no maxCombinations specified or is over MAX_SEGMENTS");
            }
        }
    }

    private Map<String, Integer> calculateFeatureSegments(List<FeatureBucketOptions> features) {
        Map<String, Integer> segments = new HashMap<>();

        for (FeatureBucketOptions feature : features) {
            if (feature.bucketOptions() == null) continue;

            if (feature.valuesCount() <= 1) {
                segments.put(feature.label(), 1);
                continue;
            }

            // For multi-value features
            Integer maxSegments = feature.maxCombinations();
            if (maxSegments == null) {
                throw new IllegalArgumentException(
                        "Feature " + feature.label() + " has multiple bucket options but no maxCombinations specified");
            }

            // Calculate how many segments we need
            int numBucketOptions = feature.bucketOptions().length;
            int numValues = feature.valuesCount();

            // Start with trying each value individually
            if (Math.pow(numBucketOptions, numValues) <= maxSegments) {
                // Can test each value individually
                segments.put(feature.label(), numValues);
            } else {
                // Need to find how many segments we can have where buckets^segments <= maxCombinations
                // log_buckets(maxCombinations) = maximum number of segments we can have
                int maxSegmentsForBuckets = (int)(Math.log(maxSegments) / Math.log(numBucketOptions));

                // Use this many segments, grouping values together
                segments.put(feature.label(), maxSegmentsForBuckets);
            }
        }

        return segments;
    }

    private long calculateTotalCombinations(List<FeatureBucketOptions> features,
                                            Map<String, Integer> segments) {
        return features.stream()
                .mapToLong(f -> {
                    if (f.bucketOptions() == null) return 1;
                    return (long)Math.pow(
                            f.bucketOptions().length,
                            segments.getOrDefault(f.label(), 1)
                    );
                })
                .reduce(1, (a, b) -> a * b);
    }

    @Override
    public synchronized List<Map<String, float[]>> getNextBatch() {
        if (currentIndex >= totalCombinations) {
            return Collections.emptyList();
        }

        List<Map<String, float[]>> batch = new ArrayList<>();
        for (long i = currentIndex;
             i < Math.min(currentIndex + batchSize, totalCombinations);
             i++) {
            batch.add(generateCombination(i));
        }

        currentIndex += batchSize;
        return batch;
    }

    @Override
    public synchronized boolean hasMoreBatches() {
        return currentIndex < totalCombinations;
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
            int bucketCount = bucketOptions.length;

            if (feature.valuesCount() <= 1) {
                // Single value feature - unchanged
                int bucketIndex = (int) (remainingIndex % bucketCount);
                result.put(feature.label(), new float[]{bucketOptions[bucketIndex]});
                remainingIndex /= bucketCount;
            } else {
                // Multi-value feature
                float[] values = new float[feature.valuesCount()];

                if (segments < feature.valuesCount()) {
                    // If we're constraining segments, treat all values as one group
                    int bucketIndex = (int) (remainingIndex % bucketCount);
                    Arrays.fill(values, bucketOptions[bucketIndex]);
                    remainingIndex /= bucketCount;
                } else {
                    // Full segmentation - each value position gets its own bucket value
                    for (int i = 0; i < feature.valuesCount(); i++) {
                        int bucketIndex = (int) ((remainingIndex / Math.pow(bucketCount, i)) % bucketCount);
                        values[i] = bucketOptions[bucketIndex];
                    }
                    remainingIndex /= Math.pow(bucketCount, feature.valuesCount());
                }

                result.put(feature.label(), values);
            }
        }

        return result;
    }

    @Override
    public void reset() {
        this.currentIndex = 0;
    }

    @Override
    public int getTotalBatches() {
        return (int)((totalCombinations + batchSize - 1) / batchSize);
    }

    public long getTotalCombinations() {
        return totalCombinations;
    }
}