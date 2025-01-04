package com.nimbus.fuzzybuckettree.tuner;

/**
 * Represents a feature to be trained on and the optional auto tuning bucket sizes to compute optimization with.
 * @param label The name of this feature used for reference, and to ensure safety in ordering of training and prediction data.
 * @param bucketOptions An optional float array of value bucketing widths used for auto tuning. For example [1, 5, 10] will
 *                      test aggregating values of this feature into groups rounded to the nearest 1, 5, or 10 to
 *                      evaluate which grouping band offering the best accuracy. This allows the decision tree to
 *                      approximate and group similar values. If null or zero length, then exact matching will be applied.
 */
public record FeatureBucketOptions(String label, int valuesCount, float[] bucketOptions, int maxCombinations) {

    /**
     * The max number of combinatorial segments this feature can have. That is, the bucketsCount^valuesCount.
     * For multi value features, testing many bucket options can quickly explode to an impractical size.
     * The auto trainer will handle automatically segmenting portions of multi value features to help
     * assist in training times and accuracy. Raising maxCombinations per bucket option will allow testing
     * of smaller segments which can be more specific, at the cost of training time and possible
     * over fitting.
     */
    public static final int MAX_SEGMENTS = 8192;

    /**
     * Construct a feature bucket options config with a default maxCombinations of 256
     * @param label
     * @param valuesCount
     * @param bucketOptions
     */
    public FeatureBucketOptions(String label, int valuesCount, float[] bucketOptions) {
        this(label, valuesCount, bucketOptions, 256);
    }

    /**
     * Construct a config used in specifying training details about a particular feature.
     * @param label The label, or id, which must be consistent. E.g. 'color'
     * @param valuesCount The number of values this feature represents, one or more. E.g. color may only be 'blue',
     *                    but timeseries data may have summary intervals in which a feature represents and their order
     *                    must be preserved.
     * @param bucketOptions An optional float array of bucket widths to use while auto tuning aggregating of feature
     *                      values. e.g. a value of 1f will auto group all values to the nearest whole integer. If
     *                      null or a zero len array, bucking is disabled and feature is exact match only. If multiple
     *                      buckets are specified, auto tuning will evaluate each bucket aggregation for optimal accuracy.
     *                      This includes if the feature is multi value, tuning will make an attempt to optimize each value
     *                      offset individually with its own bucket settings. For example, a larger aggregation for older
     *                      timeseries data.
     * @param maxCombinations Max number of combinatorial bucket options to produce while auto testing. If a feature has
     *                    8 values, and 4 bucket options, then there are 4^8=65536 combinations (segments) if each bucket, for each
     *                    value, is tested individually. This becomes impractical for training at reasonable scale.
     *                    The default is 256. The auto tuner will automatically aggregate segments of multi value features
     *                    as needed in attempt to test different bucket distributions across values, while keeping the
     *                    resulting segments under this maxCombinations count. So a lower value means much faster training
     *                    and more aggregation for value count/bucket combinations that exceed maxCombinations, and a higher
     *                    count allow the tuner to test bucket options for smaller value segments, or individually, which
     *                    can potentially produce a better result at the cost of significant additional training time.
     */
    public FeatureBucketOptions(String label, int valuesCount, float[] bucketOptions, int maxCombinations) {
        this.label = label;
        this.valuesCount = valuesCount;
        this.bucketOptions = bucketOptions;
        this.maxCombinations = maxCombinations;

        if (label == null || label.isEmpty())
            throw new IllegalArgumentException("label cannot be null");
        if (valuesCount <= 0)
            throw new IllegalArgumentException("Feature valuesCount must be greater than 0");
        if (maxCombinations <= 0 || maxCombinations > MAX_SEGMENTS)
            throw new IllegalArgumentException("maxCombinations must be greater than 0 and less than MAX_SEGMENTS " + MAX_SEGMENTS);
    }

}
