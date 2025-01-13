package com.nimbus.fuzzybuckettree;


import com.nimbus.fuzzybuckettree.prediction.NodePrediction;
import com.nimbus.fuzzybuckettree.prediction.handlers.PredictionHandler;
import com.nimbus.fuzzybuckettree.tuner.*;
import com.nimbus.fuzzybuckettree.tuner.reporters.AccuracyReporter;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;
import java.util.function.Consumer;
import java.util.stream.Collectors;

public class FuzzyBucketTuner<T> {

    private static final int NCPUS = Runtime.getRuntime().availableProcessors();

    private final ExecutorService executor;
    private final ScheduledExecutorService scheduledExecutor;
    private final List<FeatureBucketOptions> features;
    private final PredictionHandler predictionHandler;
    private final AccuracyReporter<T> accuracyReporter;
    private final AtomicReference<Float> bestAccuracy;
    private final BucketIterator bucketIterator;
    private final BucketPerformanceTracker bucketTracker;
    private final Semaphore semaphore;
    private volatile TunerResult<T> bestResult;

    public FuzzyBucketTuner(List<FeatureBucketOptions> features, PredictionHandler<T> predictionHandler,
                            AccuracyReporter<T> accuracyReporter, int concurrency) {
        this.executor = Executors.newFixedThreadPool(concurrency + 1);
        this.scheduledExecutor = Executors.newSingleThreadScheduledExecutor();
        this.features = features;
        this.predictionHandler = predictionHandler;
        this.accuracyReporter = accuracyReporter;
        this.bestAccuracy = new AtomicReference<>(0f);
        this.bucketIterator = new PatternedBucketIterator(features, 1);
        this.bucketTracker = new BucketPerformanceTracker(0.2f);
        this.semaphore = new Semaphore(Math.min(concurrency, NCPUS));
    }

    public FuzzyBucketTuner(List<FeatureBucketOptions> features, PredictionHandler<T> predictionHandler,
                            AccuracyReporter<T> accuracyReporter) {
        this(features, predictionHandler, accuracyReporter, NCPUS);
    }

    /**
     * Trains a supplied particular feature set order, and evaluates where applicable
     * every sub combination of bucket sizes
     * @param features
     * @param trainingData
     * @param validationData
     * @param skipPoorBucketConfigs true if should begin auto pruning to skip the lowest average
     *                              performing bucket configs, which can help ensure that only
     *                              the top bucket configs are tested among future feature combinations.
     *                              This can possible prevent catching the one most optimal combination
     *                              of feature and bucket ordering, but aids in reducing the search
     *                              significantly by allowing wider initial search bucket parameters.
     * @return The best single {@link TunerResult} from this feature order run
     */
    private CompletableFuture<TunerResult<T>> trainSet(List<FeatureBucketOptions> features, List<TrainingEntry<T>> trainingData,
                                 List<TrainingEntry<T>> validationData, boolean skipPoorBucketConfigs, boolean useSemaphores) {
        AtomicReference<TunerResult> bestBucketResult = new AtomicReference<>(null);
        AtomicInteger rounds = new AtomicInteger(0);

        bucketIterator.reset();
        List<CompletableFuture<Void>> futures = new ArrayList<>();

        while (bucketIterator.hasMoreBatches()) {
            List<Map<String, float[]>> batch = bucketIterator.getNextBatch();

            if (useSemaphores)
                semaphore.acquireUninterruptibly();

            CompletableFuture<Void> future = new CompletableFuture<>();
            futures.add(future);

            executor.execute(() -> {
                for (Map<String, float[]> bucketPermutationMap : batch) {
                    int bucketKey = bucketTracker.shouldTest(bucketPermutationMap);
                    if (skipPoorBucketConfigs && bucketKey == 0)
                        continue;

                    rounds.incrementAndGet();

                    List<FeatureConfig> featureConfigs = features.stream()
                            .map(f -> new FeatureConfig(f.label(), f.type(), f.valuesCount(), bucketPermutationMap.get(f.label())))
                            .toList();

                    FuzzyBucketTree<T> tree = new FuzzyBucketTree<>(featureConfigs, predictionHandler.newHandlerInstance());

                    for (TrainingEntry<T> trainingEntry : trainingData) {
                        tree.train(trainingEntry.features(), trainingEntry.outcome());
                    }

                    AccuracyReporter<T> accuracyReporter = this.accuracyReporter.getNewInstance();

                    for (TrainingEntry<T> validationEntry : validationData) {
                        NodePrediction<T> prediction = tree.predict(validationEntry.features());
                        accuracyReporter.record(prediction.getPrediction(), validationEntry.outcome());
                    }

                    TunerResult<T> result = new TunerResult<>(featureConfigs, accuracyReporter, null);

                    if (bucketKey > 0)
                        bucketTracker.recordPerformance(bucketKey, result.getTotalAccuracy());

                    synchronized (bestBucketResult) {
                        if (bestBucketResult.get() == null ||
                                bestBucketResult.get().getTotalAccuracy() < result.getTotalAccuracy()) {
                            bestBucketResult.set(result);
                        }
                    }
                }

                if (useSemaphores)
                    semaphore.release();
                future.complete(null);
            });
        }

        return CompletableFuture.allOf(futures.toArray(new CompletableFuture[0])).thenApply(v -> bestBucketResult.get());
    }

    private ScheduledFuture<?> startMonitor(AtomicInteger complete, int total) {
        AtomicLong startTime = new AtomicLong(System.currentTimeMillis());

        return scheduledExecutor.scheduleAtFixedRate(() -> {
            int done = complete.get();
            float percent = (done / (float) total) * 100f;

            long elapsedMs = System.currentTimeMillis() - startTime.get();
            long elapsedSecs = elapsedMs / 1000;

            // Calculate rate of completion (jobs per second)
            double jobsPerSecond = done / (elapsedMs / 1000.0);

            // Estimate remaining time
            int remainingJobs = total - done;
            long estimatedSecsRemaining = jobsPerSecond > 0 ?
                    (long)(remainingJobs / jobsPerSecond) : 0;

            System.out.printf("TUNE Jobs complete: %d of %d (%.1f%%) | " +
                            "Elapsed: %02d:%02d:%02d | " +
                            "Estimated remaining: %02d:%02d:%02d | " +
                            "Rate: %.1f jobs/sec | " +
                            "Best Accuracy %.3f%%%n\n",
                    done, total, percent,
                    elapsedSecs / 3600, (elapsedSecs % 3600) / 60, elapsedSecs % 60,
                    estimatedSecsRemaining / 3600, (estimatedSecsRemaining % 3600) / 60, estimatedSecsRemaining % 60,
                    jobsPerSecond,
                    bestAccuracy.get()
            );
        }, 1, 1, TimeUnit.SECONDS);
    }

    public CompletableFuture<Void> submitDataBatches(List<TrainingEntry<T>> data, Consumer<TrainingEntry<T>> job) {
        int numThreads = Runtime.getRuntime().availableProcessors();
        int totalSize = data.size();
        int batchSize = (totalSize + numThreads - 1) / numThreads;

        CountDownLatch latch = new CountDownLatch(numThreads);
        AtomicReference<Throwable> error = new AtomicReference<>();

        for (int threadIndex = 0; threadIndex < numThreads; threadIndex++) {
            final int start = threadIndex * batchSize;
            final int end = Math.min(start + batchSize, totalSize);

            // Skip if this thread would have no data to process
            if (start >= totalSize) {
                latch.countDown();
                continue;
            }

            semaphore.acquireUninterruptibly();
            executor.execute(() -> {
                try {
                    for (int i = start; i < end; i++) {
                        job.accept(data.get(i));
                    }
                } catch (Throwable t) {
                    error.set(t);
                } finally {
                    semaphore.release();
                    latch.countDown();
                }
            });
        }

        return CompletableFuture.runAsync(() -> {
            try {
                latch.await();
                if (error.get() != null) {
                    throw new CompletionException(error.get());
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new CompletionException(e);
            }
        });
    }

    public static <K> List<K> sampleList(List<K> input, float sampleRate) {
        if (sampleRate >= 1.0f) return input;
        if (sampleRate <= 0.0f) throw new IllegalArgumentException("Sample rate must be > 0");

        int stride = Math.round(1.0f / sampleRate);
        List<K> result = new ArrayList<>(input.size() / stride + 1);

        for (int i = 0; i < input.size(); i += stride) {
            result.add(input.get(i));
        }

        return result;
    }

    /**
     * Begin the full training and auto-tuning process, which will evaluate every
     * possible permutation of provided feature order as well as bucket delta values for
     * each individual feature as configured. Features which return more than one value
     * will always have their order respected, and will not be shuffled e.g. time
     * series data. This reviews <b>all</b> training data, and <b>all</b> feature
     * permutations. If your dataset or bucketing options are large, see the other
     * training method which is magnitudes quicker through data sampling during auto
     * tuning and early stopping detection.
     * @param trainingData Data to train on as a list of individual map entries,
     *             mapping each feature to their respective value(s).
     *             The map must contain a <i>target</i> key which contains
     *             the target prediction value for the given feature set.
     * @param validationData Data set used solely for accurady validation and not training.
     *                       Must also include the <i>target</i> value.
     * @implNote This trains and validates on 100% of the supplied data which if the dataset is large
     * or there are many bucketing options to tune, can be impractically large. Use the other train
     * method which accepts a sample rate for the auto tuning stages to determine optimal bucketing
     * before completing a full final train.
     * @return {@link TunerResult} with the best configration and accuracy results found.
     */
    public TunerResult<T> train(List<TrainingEntry<T>> trainingData, List<TrainingEntry<T>> validationData) {
        return trainAllPermutations(trainingData, validationData, 1f, 1f);
    }

    /**
     * Begin the full training and auto-tuning process, which will evaluate every
     * possible permutation of provided feature order as well as bucket delta values for
     * each individual feature as configured. Features which return more than one value
     * will always have their order respected, and will not be shuffled e.g. time
     * series data. This method will sample data during auto optimization to accelerate
     * bucket parameter exploration. Once the best config is found, a full training
     * and validation will be ran for the final returned tree model. This is the
     * slowest training method but has the most coverage of feature order evaluation.
     * @param trainingData Data to train on as a list of individual map entries,
     *             mapping each feature to their respective value(s).
     *             The map must contain a <i>target</i> key which contains
     *             the target prediction value for the given feature set.
     * @param validationData Data set used solely for accurady validation and not training.
     *                       Must also include the <i>target</i> value.
     * @param sampleRate A sample rate percentage between 0 and 1. For example 0.2 would be sample of 20%
     *                   of supplied data used for training and validation during the auto-tuning phases, which
     *                   can take very long. 1f means no sampling.
     * @param earlyStoppingPercent A value between 0 and 1 which is a percentage of limit of feature permutations
     *                             to exit auto tuning early if no improvement is found. E.g. if there are 5,000
     *                             feature permutations and an early stopping of 0.1f, the tuning will end early
     *                             if no improvement is found in the last 500 permutations. 1f means no early stopping.
     * @return {@link TunerResult} with the best configration and accuracy results found.
     */
    public TunerResult<T> trainAllPermutations(List<TrainingEntry<T>> trainingData, List<TrainingEntry<T>> validationData,
                                float sampleRate, float earlyStoppingPercent) {
        if (trainingData.isEmpty() || validationData.isEmpty())
            throw new IllegalArgumentException("Training and validation data must not be empty");

        if (sampleRate <= 0f || sampleRate > 1.0f)
            throw new IllegalArgumentException("Sample rate must be between greater than 0 and less or equal to 1");

        if (earlyStoppingPercent <= 0f || earlyStoppingPercent > 1.0f)
            throw new IllegalArgumentException("Early stopping cannot 0 or greeter than 1");

        Map<String, FeatureBucketOptions> featureBucketCache = features.stream().collect(
                Collectors.toMap(FeatureBucketOptions::label, f -> f));

        List<List<String>> permutations = TunerUtils.generateRandomPermutations(
                new ArrayList<>(featureBucketCache.keySet()));
        int bucketMultiplier = Math.max(1, features.stream()
                .mapToInt(f -> f.bucketOptions() != null && f.bucketOptions().length > 0 ? f.bucketOptions().length - 1 : 0)
                .sum());
        int totalRounds = permutations.size() * bucketMultiplier;
        int earlyStoppingJobs = earlyStoppingPercent < 1f ? (int)(permutations.size() * earlyStoppingPercent) : 0;

        if (totalRounds <= 0 || totalRounds == Integer.MAX_VALUE)
            throw new IllegalArgumentException("Too few or too many feature combinations provided, calculation impractical!");

        List<TrainingEntry<T>> trainingSample = sampleRate < 1f ? sampleList(trainingData, sampleRate) : trainingData;
        List<TrainingEntry<T>> validationSample = sampleRate < 1f ? sampleList(validationData, sampleRate) : validationData;

        System.out.println("Begin processing " + trainingSample.size() + " rows of sampled data across "
                + permutations.size() + " feature permutations");

        AtomicInteger complete = new AtomicInteger();
        AtomicInteger jobsSinceImprovement = new AtomicInteger();

        ScheduledFuture statusFut = startMonitor(complete, permutations.size());

        boolean mainLoopSemaphore = bucketMultiplier == 1;
        CountDownLatch latch = new CountDownLatch(1);
        AtomicBoolean shouldStop = new AtomicBoolean(false);
        for (List<String> featureOrderSet : permutations) {
            if (shouldStop.get())
                break;

            if (mainLoopSemaphore)
                semaphore.acquireUninterruptibly();

            CompletableFuture<Void> fut = new CompletableFuture<>();
            executor.execute(() -> {
                try {
                    List<FeatureBucketOptions> featureBucketPermutation = featureOrderSet.stream().map(
                            f -> featureBucketCache.get(f)).toList();

                    // We begin testing all bucket configs up until sampleRate coverage of feature permutations,
                    // then from there we only continue testing the top bucket option performers. This allows
                    // us to begin with a wide search field of bucket options, but quickly narrow it down to
                    // reduce wasted training time on bucket options that consistently perform poorly
                    boolean skipLowBucketPerformers = bucketMultiplier > NCPUS && (complete.get() / (float) permutations.size()) > sampleRate;

                    TunerResult<T> res = trainSet(featureBucketPermutation,
                            trainingSample,
                            validationSample,
                            skipLowBucketPerformers,
                            !mainLoopSemaphore).get();

                    if (res == null)
                        throw new RuntimeException("Bucket permutation trainSet call returned null TunerResult. Panic!");

                    float resAccuracy = res.getTotalAccuracy();
                    float bestSoFar = bestAccuracy.getAndUpdate(curAccuracy
                            -> resAccuracy > curAccuracy ? resAccuracy : curAccuracy);

                    if (bestResult == null || resAccuracy > bestSoFar) {
                        bestResult = res;
                        jobsSinceImprovement.set(0);
                    } else if (earlyStoppingJobs > 0 && jobsSinceImprovement.incrementAndGet() > earlyStoppingJobs) {
                        System.out.println("No improvement in " + earlyStoppingJobs + ", exiting auto-tuning early");
                        latch.countDown();
                        shouldStop.set(true);
                        return;
                    }

                    if (complete.incrementAndGet() == permutations.size()) {
                        latch.countDown();
                        System.out.println("Finished all permutations");
                        shouldStop.set(true);
                        return;
                    }

                    if (res.getTotalAccuracy() == 1f && latch.getCount() == 1) {
                        System.out.println("Early exit, found perfect result");
                        latch.countDown();
                        shouldStop.set(true);
                    }
                } catch (Throwable e) {
                    latch.countDown();
                    System.out.println(e.getMessage());
                    e.printStackTrace();
                    System.exit(-1);
                } finally {
                    fut.complete(null);
                }
            });

            try {
                fut.get();
            } catch (Throwable e) {
                throw new RuntimeException(e);
            }

            if (mainLoopSemaphore)
                semaphore.release();
        }

        try {
            latch.await();
            statusFut.cancel(true);

            System.out.println("Initial auto-tuning complete! Best sampled training accuracy of " + bestResult.getTotalAccuracy()
                    + " with a feature configuration of:");
            bestResult.getFeatureConfigs().forEach(fc -> {
                System.out.println(fc.label()  + "(" + fc.valueCount() + ") -> " + Arrays.toString(fc.buckets()));
            });

            System.out.println("Now begin full data training with discovered settings..");

            FuzzyBucketTree<T> finalTree = new FuzzyBucketTree<>(bestResult.getFeatureConfigs(), predictionHandler.newHandlerInstance());

            this.submitDataBatches(trainingData, entry -> finalTree.train(entry.features(), entry.outcome())).get();

            System.out.println("Initial training done, begin validation work..");

            AccuracyReporter<T> reporter = accuracyReporter.getNewInstance();
            this.submitDataBatches(validationData, entry -> {
                NodePrediction<T> nodePrediction = finalTree.predict(entry.features());
                if (nodePrediction != null && nodePrediction.getConfidence() > 0f)
                    reporter.record(nodePrediction.getPrediction(), entry.outcome());
            }).get();

            System.out.println("Training complete. Final weighted accuracy of "
                    + new BigDecimal(reporter.getTotalAccuracy()).setScale(2, RoundingMode.HALF_EVEN));

            return new TunerResult<>(bestResult.getFeatureConfigs(), reporter, finalTree);
        } catch (Throwable e) {
            throw new RuntimeException(e);
        } finally {
            executor.shutdownNow();
            scheduledExecutor.shutdownNow();
        }
    }

    private List<TrainingEntry<T>> reduceTrainingFields(List<TrainingEntry<T>> data, List<FeatureBucketOptions> testStack) {
        return data.stream()
                .map(entry -> {
                    Set<String> allowedKeys = testStack.stream()
                            .map(FeatureBucketOptions::label)
                            .collect(Collectors.toSet());

                    Map<String, Object[]> filteredFeatures = entry.features().entrySet().stream()
                            .filter(e -> allowedKeys.contains(e.getKey()))
                            .collect(Collectors.toMap(
                                    Map.Entry::getKey,
                                    e -> e.getValue()
                            ));

                    return new TrainingEntry<>(filteredFeatures, entry.outcome());
                })
                .collect(Collectors.toList());
    }

    /**
     * Begin the training and auto-tuning process. This method seeks to reduce the
     * search space by iteratively testing and ordering features from lowest to highest
     * accuracy, in effect expecting broadest to most specific ordering.
     * Features which return more than one value will always have their order respected,
     * and will not be shuffled e.g. time series data. Once the best config is found, a full training
     * and validation will be ran for the final returned tree model.
     * @param trainingData Data to train on as a list of individual map entries,
     *             mapping each feature to their respective value(s).
     *             The map must contain a <i>target</i> key which contains
     *             the target prediction value for the given feature set.
     * @param validationData Data set used solely for accurady validation and not training.
     *                       Must also include the <i>target</i> value.
     * @implNote Temporary evaluation method - will assess bucket values but not in respect to
     * very possible combination alongside categorical features.
     * @return {@link TunerResult} with the best configration and accuracy results found.
     */
    public TunerResult<T> trainIterative(List<TrainingEntry<T>> trainingData, List<TrainingEntry<T>> validationData) {
        if (trainingData.isEmpty() || validationData.isEmpty())
            throw new IllegalArgumentException("Training and validation data must not be empty");

        int bucketMultiplier = Math.max(1, features.stream()
                .mapToInt(f -> f.bucketOptions() != null && f.bucketOptions().length > 0 ? f.bucketOptions().length - 1 : 0)
                .sum());

        int totalRounds = (features.size() * (features.size() + 1)) / 2;

        if (totalRounds <= 0 || totalRounds == Integer.MAX_VALUE)
            throw new IllegalArgumentException("Too few or too many feature combinations provided, calculation impractical!");

        AtomicInteger complete = new AtomicInteger();

        ScheduledFuture statusFut = startMonitor(complete, totalRounds);

        boolean mainLoopSemaphore = bucketMultiplier == 1;
        CountDownLatch latch = new CountDownLatch(1);
        AtomicBoolean shouldStop = new AtomicBoolean(false);

        List<FeatureStackScore> stackScores = new ArrayList<>();

        List<FeatureBucketOptions> featureStack = new ArrayList<>(features.size());
        List<FeatureBucketOptions> featuresRemaining = new ArrayList<>(features);
        while (!featuresRemaining.isEmpty()) {
            NavigableMap<Float, FeatureBucketOptions> featureTestScores = Collections.synchronizedNavigableMap(new TreeMap<>());

            Map<FeatureBucketOptions, TunerResult<T>> featureTunerResult = new ConcurrentHashMap<>();
            featuresRemaining.forEach(feature -> {
                if (mainLoopSemaphore)
                    semaphore.acquireUninterruptibly();
                if (shouldStop.get())
                    return;

                List<FeatureBucketOptions> testStack = new ArrayList<>(featureStack);
                testStack.add(feature);

                List<TrainingEntry<T>> stackTrainingData = reduceTrainingFields(trainingData, testStack);
                List<TrainingEntry<T>> stackValidationData = reduceTrainingFields(validationData, testStack);

                CompletableFuture<Void> fut = new CompletableFuture<>();
                executor.execute(() -> {
                    try {
                        // We begin testing all bucket configs up until sampleRate coverage of feature permutations,
                        // then from there we only continue testing the top bucket option performers. This allows
                        // us to begin with a wide search field of bucket options, but quickly narrow it down to
                        // reduce wasted training time on bucket options that consistently perform poorly
                        boolean skipLowBucketPerformers = false; // bucketMultiplier > NCPUS && (complete.get() / (float) totalRounds) > sampleRate;

                        TunerResult<T> res = trainSet(testStack,
                                stackTrainingData,
                                stackValidationData,
                                skipLowBucketPerformers,
                                !mainLoopSemaphore).get();

                        if (res == null)
                            throw new RuntimeException("Bucket permutation trainSet call returned null TunerResult. Panic!");

                        featureTestScores.put(res.getTotalAccuracy(), feature);
                        featureTunerResult.put(feature, res);
                    } catch (Throwable e) {
                        latch.countDown();
                        System.out.println(e.getMessage());
                        e.printStackTrace();
                        System.exit(-1);
                    } finally {
                        fut.complete(null);
                    }
                });

                try {
                    fut.get();
                } catch (Throwable e) {
                    throw new RuntimeException(e);
                }

                if (mainLoopSemaphore)
                    semaphore.release();

                complete.incrementAndGet();
            });

            Map.Entry<Float, FeatureBucketOptions> worstEntry = featureTestScores.pollFirstEntry();
            Map.Entry<Float, FeatureBucketOptions> bestEntry = featureTestScores.pollLastEntry();
            if (bestEntry == null) // same scores
                bestEntry = worstEntry;

            stackScores.add(new FeatureStackScore(worstEntry.getValue().label(), worstEntry.getKey()));
            featureStack.add(worstEntry.getValue());
            featuresRemaining.remove(worstEntry.getValue());

            TunerResult<T> bestFeatureResult = featureTunerResult.get(bestEntry.getValue());

            /*
            if (bestResult != null && bestResult.getTotalAccuracy() > bestFeatureResult.getTotalAccuracy()) {
                System.out.println("All remaining features made model worse or provided no improvement, early stopping");
                shouldStop.set(true);
                break;
            } */

            bestAccuracy.set(bestFeatureResult.getTotalAccuracy());
            bestResult = bestFeatureResult;

            System.out.println("Next identified feature " + worstEntry.getValue().label() + " score " + worstEntry.getKey());
        }

        try {
            statusFut.cancel(true);

            System.out.println("Initial auto-tuning complete! Best sampled training accuracy of " + bestResult.getTotalAccuracy()
                    + " with a feature configuration of:");
            bestResult.getFeatureConfigs().forEach(fc -> {
                System.out.println(fc.label()  + "(" + fc.valueCount() + ") -> " + Arrays.toString(fc.buckets()));
            });
            System.out.println("Feature score progression: " + stackScores.stream().map(s
                    -> s.feature() + " (" + s.score() + ")").collect(Collectors.joining(" -> ")));

            System.out.println("Now begin full data training with discovered settings..");

            FuzzyBucketTree<T> finalTree = new FuzzyBucketTree<>(bestResult.getFeatureConfigs(), predictionHandler.newHandlerInstance());

            this.submitDataBatches(trainingData, entry -> finalTree.train(entry.features(), entry.outcome())).get();

            System.out.println("Initial training done, begin validation work..");

            AccuracyReporter<T> reporter = accuracyReporter.getNewInstance();
            this.submitDataBatches(validationData, entry -> {
                NodePrediction<T> nodePrediction = finalTree.predict(entry.features());
                if (nodePrediction != null && nodePrediction.getConfidence() > 0f)
                    reporter.record(nodePrediction.getPrediction(), entry.outcome());
            }).get();

            System.out.println("Training complete. Final weighted accuracy of "
                    + new BigDecimal(reporter.getTotalAccuracy()).setScale(2, RoundingMode.HALF_EVEN));

            return new TunerResult<>(bestResult.getFeatureConfigs(), reporter, finalTree);
        } catch (Throwable e) {
            throw new RuntimeException(e);
        } finally {
            executor.shutdownNow();
            scheduledExecutor.shutdownNow();
        }
    }

    /**
     * Begin the training and auto-tuning process. This method seeks to reduce the
     * search space by iteratively testing and ordering categorical features not by
     * individual accuracy, but by highest to lowest average frequency. E.g.
     * broad features that have low cardinality and high sample rates will end up
     * shallower in the tree, and lower ones deeper. This is an alternative approach
     * to finding the most optimal hierarchical node order.
     * After identified, combinations of these categorical feature orders are tested
     * separately with different orderings and configurations of the bucketed features to
     * determine the best placement and bucketing of applicable features when combined with
     * the exact match feature set.
     * Features which return more than one value will always have their order respected,
     * and will not be shuffled e.g. time series data. Once the best config is found, a full training
     * and validation will be ran for the final returned tree model.
     * @param trainingData Data to train on as a list of individual map entries,
     *             mapping each feature to their respective value(s).
     *             The map must contain a <i>target</i> key which contains
     *             the target prediction value for the given feature set.
     * @param validationData Data set used solely for accurady validation and not training.
     *                       Must also include the <i>target</i> value.
     * @implNote Temporary evaluation method - will assess bucket values but not in respect to
     * very possible combination alongside categorical features.
     * @return {@link TunerResult} with the best configration and accuracy results found.
     */
    public TunerResult<T> trainFrequency(List<TrainingEntry<T>> trainingData, List<TrainingEntry<T>> validationData) throws ExecutionException, InterruptedException {
        if (trainingData.isEmpty() || validationData.isEmpty())
            throw new IllegalArgumentException("Training and validation data must not be empty");

        // Split features into exact match and bucketed
        List<FeatureBucketOptions> exactMatchFeatures = features.stream()
                .filter(f -> f.bucketOptions() == null || f.bucketOptions().length == 0)
                .collect(Collectors.toList());

        List<FeatureBucketOptions> bucketedFeatures = features.stream()
                .filter(f -> f.bucketOptions() != null && f.bucketOptions().length > 0)
                .collect(Collectors.toList());

        // Calculate bucket multiplier for parallelization decision
        int bucketMultiplier = Math.max(1, bucketedFeatures.stream()
                .mapToInt(f -> f.bucketOptions().length - 1)
                .sum());

        // Calculate frequencies and sort exact match features
        Map<FeatureBucketOptions, Double> frequencies = calculateExactMatchFrequencies(trainingData, exactMatchFeatures);
        List<FeatureBucketOptions> orderedExactFeatures = exactMatchFeatures.stream()
                .sorted((f1, f2) -> Double.compare(frequencies.get(f2), frequencies.get(f1)))
                .collect(Collectors.toList());

        // Generate positions for bucketed features
        List<List<Integer>> bucketPositions = generateBucketPositions(
                orderedExactFeatures.size(),
                bucketedFeatures.size()
        );

        int totalCombinations = bucketPositions.size();
        if (totalCombinations == 0 || totalCombinations == Integer.MAX_VALUE)
            throw new IllegalArgumentException("Too few or too many feature combinations!");

        AtomicInteger complete = new AtomicInteger();
        ScheduledFuture statusFut = startMonitor(complete, totalCombinations);

        boolean mainLoopSemaphore = bucketMultiplier == 1;
        CountDownLatch latch = new CountDownLatch(1);
        AtomicBoolean shouldStop = new AtomicBoolean(false);
        AtomicReference<TunerResult<T>> bestResult = new AtomicReference<>();
        AtomicReference<List<FeatureBucketOptions>> bestFeatureOrder = new AtomicReference<>();
        AtomicReference<Float> bestAccuracy = new AtomicReference<>(0f);

        for (List<Integer> positions : bucketPositions) {
            if (shouldStop.get())
                break;

            if (mainLoopSemaphore)
                semaphore.acquireUninterruptibly();

            CompletableFuture<Void> fut = new CompletableFuture<>();
            executor.execute(() -> {
                try {
                    // Create feature order for this combination
                    List<FeatureBucketOptions> testFeatures = new ArrayList<>(orderedExactFeatures);
                    for (int i = 0; i < bucketedFeatures.size(); i++) {
                        testFeatures.add(positions.get(i), bucketedFeatures.get(i));
                    }

                    // Test this feature order
                    TunerResult<T> result = trainSet(
                            testFeatures,
                            trainingData,
                            validationData,
                            false,
                            !mainLoopSemaphore  // Let trainSet handle parallelization when we have multiple bucket options
                    ).get();

                    if (result == null)
                        throw new RuntimeException("Bucket permutation trainSet call returned null TunerResult!");

                    float resAccuracy = result.getTotalAccuracy();
                    float currentBest = bestAccuracy.get();

                    if (resAccuracy > currentBest) {
                        bestAccuracy.set(resAccuracy);
                        bestResult.set(result);
                        bestFeatureOrder.set(new ArrayList<>(testFeatures));

                        System.out.println("New best accuracy: " + resAccuracy + " with feature order: " +
                                testFeatures.stream().map(FeatureBucketOptions::label)
                                        .collect(Collectors.joining(" -> ")));
                    }

                    if (complete.incrementAndGet() == totalCombinations) {
                        latch.countDown();
                        System.out.println("Finished all combinations");
                        shouldStop.set(true);
                        return;
                    }

                    if (resAccuracy == 1f && latch.getCount() == 1) {
                        System.out.println("Early exit, found perfect result");
                        latch.countDown();
                        shouldStop.set(true);
                    }
                } catch (Throwable e) {
                    latch.countDown();
                    System.out.println(e.getMessage());
                    e.printStackTrace();
                    System.exit(-1);
                } finally {
                    fut.complete(null);
                }
            });

            try {
                fut.get();
            } catch (Throwable e) {
                throw new RuntimeException(e);
            }

            if (mainLoopSemaphore)
                semaphore.release();
        }

        try {
            latch.await();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("Interrupted while waiting for completion", e);
        }

        statusFut.cancel(true);

        return finalizeTuning(bestResult.get(), bestFeatureOrder.get(), trainingData, validationData);
    }

    private Map<FeatureBucketOptions, Double> calculateExactMatchFrequencies(
            List<TrainingEntry<T>> trainingData,
            List<FeatureBucketOptions> exactMatchFeatures) {

        // Use a more memory-efficient approach with parallel processing
        Map<FeatureBucketOptions, Map<Object, LongAdder>> valueCounts = new ConcurrentHashMap<>();
        exactMatchFeatures.forEach(f -> valueCounts.put(f, new ConcurrentHashMap<>()));

        // Process in batches to reduce memory pressure
        int batchSize = 10000;
        for (int i = 0; i < trainingData.size(); i += batchSize) {
            int endIndex = Math.min(i + batchSize, trainingData.size());
            List<TrainingEntry<T>> batch = trainingData.subList(i, endIndex);

            batch.parallelStream().forEach(entry -> {
                for (FeatureBucketOptions feature : exactMatchFeatures) {
                    Object value = entry.features().get(feature.label());
                    if (value != null) {
                        valueCounts.get(feature)
                                .computeIfAbsent(value, k -> new LongAdder())
                                .increment();
                    }
                }
            });
        }

        // Calculate frequencies
        Map<FeatureBucketOptions, Double> frequencies = new HashMap<>();
        for (FeatureBucketOptions feature : exactMatchFeatures) {
            Map<Object, LongAdder> counts = valueCounts.get(feature);
            long totalSamples = counts.values().stream()
                    .mapToLong(LongAdder::sum)
                    .sum();
            double uniqueValues = counts.size();
            frequencies.put(feature, (double) totalSamples / uniqueValues);
        }

        return frequencies;
    }

    private List<List<Integer>> generateBucketPositions(int fixedFeatureCount, int bucketedFeatureCount) {
        List<List<Integer>> positions = new ArrayList<>();

        // Generate all possible positions for bucketed features
        int[] currentPositions = new int[bucketedFeatureCount];
        for (int i = 0; i < bucketedFeatureCount; i++) {
            currentPositions[i] = i;
        }

        do {
            List<Integer> positionList = Arrays.stream(currentPositions)
                    .boxed()
                    .collect(Collectors.toList());
            positions.add(positionList);
        } while (generateNextPosition(currentPositions, fixedFeatureCount + bucketedFeatureCount - 1));

        return positions;
    }

    private boolean generateNextPosition(int[] positions, int maxValue) {
        int i = positions.length - 1;
        while (i >= 0 && positions[i] == maxValue - (positions.length - 1 - i)) {
            i--;
        }

        if (i < 0) {
            return false;
        }

        positions[i]++;
        for (int j = i + 1; j < positions.length; j++) {
            positions[j] = positions[j - 1] + 1;
        }
        return true;
    }

    private TunerResult<T> finalizeTuning(
            TunerResult<T> bestResult,
            List<FeatureBucketOptions> bestFeatureOrder,
            List<TrainingEntry<T>> trainingData,
            List<TrainingEntry<T>> validationData) throws ExecutionException, InterruptedException {

        System.out.println("Auto-tuning complete! Best accuracy: " + bestResult.getTotalAccuracy());
        System.out.println("Final feature order:");
        bestResult.getFeatureConfigs().forEach(f -> {
            String bucketInfo = f.buckets() != null && f.buckets().length > 0 ?
                    " (bucketed: " + Arrays.toString(f.buckets()) + ")" :
                    " (exact match)";
            System.out.println("- " + f.label() + bucketInfo);
        });

        // Train final model with best configuration
        FuzzyBucketTree<T> finalTree = new FuzzyBucketTree<>(
                bestResult.getFeatureConfigs(),
                predictionHandler.newHandlerInstance()
        );

        this.submitDataBatches(trainingData,
                entry -> finalTree.train(entry.features(), entry.outcome())).get();

        AccuracyReporter<T> reporter = accuracyReporter.getNewInstance();
        this.submitDataBatches(validationData, entry -> {
            NodePrediction<T> prediction = finalTree.predict(entry.features());
            if (prediction != null && prediction.getConfidence() > 0f) {
                reporter.record(prediction.getPrediction(), entry.outcome());
            }
        }).get();

        return new TunerResult<>(bestResult.getFeatureConfigs(), reporter, finalTree);
    }

}
