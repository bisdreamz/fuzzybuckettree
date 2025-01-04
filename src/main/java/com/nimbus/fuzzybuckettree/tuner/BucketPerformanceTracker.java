package com.nimbus.fuzzybuckettree.tuner;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.Collectors;

public class BucketPerformanceTracker {
    private final double retentionThreshold;
    private final Map<Integer, BucketConfigPerformance> performanceMap = new ConcurrentHashMap<>();
    private final NavigableMap<Double, Set<Integer>> sortedPerformances = Collections.synchronizedNavigableMap(new TreeMap<>());
    private volatile int allowedBucketCount = Integer.MAX_VALUE;  // Start allowing all until we have enough data

    public BucketPerformanceTracker(double retentionThreshold) {
        this.retentionThreshold = retentionThreshold;
    }

    public class BucketConfigPerformance {
        private final AtomicReference<Double> bestAccuracy = new AtomicReference<>(Double.NEGATIVE_INFINITY);

        public double getBestAccuracy() {
            return bestAccuracy.get();
        }

        public void recordAccuracy(float accuracy) {
            bestAccuracy.updateAndGet(current -> Math.max(current, accuracy));
        }
    }

    public synchronized void recordPerformance(int key, float accuracy) {
        BucketConfigPerformance perf = performanceMap.computeIfAbsent(key,
                k -> new BucketConfigPerformance());

        // Remove old best if it exists
        double oldBest = perf.getBestAccuracy();
        if (!Double.isNaN(oldBest)) {
            Set<Integer> configs = sortedPerformances.get(oldBest);
            if (configs != null) {
                configs.remove(key);
                if (configs.isEmpty()) {
                    sortedPerformances.remove(oldBest);
                }
            }
        }

        // Add new performance data
        perf.recordAccuracy(accuracy);

        // Add new best
        double newBest = perf.getBestAccuracy();
        sortedPerformances.computeIfAbsent(newBest, k -> new HashSet<>())
                .add(key);

        // Update allowed bucket count - hard cutoff at X% of total unique buckets
        allowedBucketCount = Math.max(1, (int)(performanceMap.size() * retentionThreshold));
    }

    private int getBucketConfigHash(Map<String, float[]> config) {
        return config.entrySet().stream()
                .sorted(Map.Entry.comparingByKey())
                .mapToInt(e -> 31 * e.getKey().hashCode() + Arrays.hashCode(e.getValue()))
                .reduce(0, (a, b) -> 31 * a + b);
    }

    /**
     * @param bucketConfig Map of the feature -> bucket config to check if we should test.
     *                     A bucket config passes if it is in the top <i>retentionThreshold</i>
     *                     as passed in the constructor, or if it has not been seen yet.
     * @return A non-zero long value which is a key to the bucket config for calling {@link #recordPerformance(int, float)},
     * or a value of 0 which indicates this config should be skipped
     */
    public int shouldTest(Map<String, float[]> bucketConfig) {
        int key = getBucketConfigHash(bucketConfig);
        BucketConfigPerformance perf = performanceMap.get(key);

        if (perf == null) return key;  // Always test new configs

        // Take snapshot of data we need inside synchronized block
        NavigableMap<Double, Integer> sizesByAccuracy;
        int currentAllowed;
        synchronized(this) {
            sizesByAccuracy = sortedPerformances.entrySet().stream()
                    .collect(Collectors.toMap(
                            Map.Entry::getKey,
                            e -> e.getValue().size(),
                            (a, b) -> a,  // Shouldn't happen
                            TreeMap::new
                    ));
            currentAllowed = allowedBucketCount;
        }

        // Calculate rank using snapshot
        int rank = 0;
        double bestAcc = perf.getBestAccuracy();
        for (Map.Entry<Double, Integer> entry : sizesByAccuracy.descendingMap().entrySet()) {
            if (entry.getKey() <= bestAcc) break;
            rank += entry.getValue();
        }

        boolean shouldTest = rank < currentAllowed ||
                ThreadLocalRandom.current().nextInt(100) < 5;

        return shouldTest ? key : 0;
    }
}