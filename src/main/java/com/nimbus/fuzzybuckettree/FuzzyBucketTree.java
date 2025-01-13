package com.nimbus.fuzzybuckettree;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.nimbus.fuzzybuckettree.prediction.NodePrediction;
import com.nimbus.fuzzybuckettree.prediction.Prediction;
import com.nimbus.fuzzybuckettree.prediction.handlers.PredictionHandler;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

public class FuzzyBucketTree<T> {

    private static final ScheduledExecutorService EXECUTOR = Executors.newScheduledThreadPool(2, r -> {
        Thread thread = new Thread(r);
        thread.setDaemon(true);
        thread.setName("FuzzyTreeCleaner");
        return thread;
    });

    private static final NodePrediction NO_PRED = new NodePrediction<>(new Prediction<>(null, 0f), 0);

    public static <T> FuzzyBucketTree<T> loadModel(String path) throws IOException {
        try (InputStream inputStream = Files.newInputStream(Paths.get(path))) {
            return ModelObjectMapper.MAPPER.readValue(inputStream,
                    ModelObjectMapper.MAPPER.getTypeFactory().constructParametricType(FuzzyBucketTree.class, Object.class));
        } catch (IOException e) {
            throw new IOException("Failed to load model from " + path, e);
        }
    }

    @JsonProperty("root")
    private final FeatureNode<?, T> root;
    @JsonProperty("features")
    private final List<FeatureConfig> features;
    @JsonProperty("pruning_window")
    private Duration pruningWindow;
    private ScheduledFuture cleanerTask;

    @JsonCreator
    public FuzzyBucketTree(
            @JsonProperty("root") FeatureNode<?, T> root,
            @JsonProperty("features") List<FeatureConfig> features,
            @JsonProperty("pruning_window") Duration pruningWindow) {
        this.root = root;
        this.features = features;
        this.pruningWindow = pruningWindow;
        updateCleanerTask(pruningWindow);
    }

    /**
     * Construct a FuzzyBucketTree with the associated features and prediction handler. In most cases,
     * this is the result of an auto-tuning exploration with {@link FuzzyBucketTuner}
     * @param features An ordered list of {@link FeatureConfig} indicating the features the tree will comprise,
     *                 the per feature value count, and finally the per feature bucket value if applicable.
     * @param predictionHandler A {@link PredictionHandler} instance which is respnsible for returning the
     *                          prediction results from a leaf note. Common ones in {@link com.nimbus.fuzzybuckettree.prediction.PredictionHandlers}
     *                          or users may implement their own custom prediction logic, e.g. if a minimum number of
     *                          samples is required to consider a valid result.
     * @param pruningWindow A duration which if positive will prune leafs and tree branches
     *                      when determined stale by the {@link PredictionHandler#shouldPrune()}. For
     *                      implementing data expiry in an online training environment.
     */
    public FuzzyBucketTree(List<FeatureConfig> features, PredictionHandler<T> predictionHandler, Duration pruningWindow) {
        this.features = features;

        if (features == null || features.isEmpty())
            throw new IllegalArgumentException("Features cannot be empty");


        Map<String, FeatureConfig> featureCache = features.stream().collect(Collectors.toMap(FeatureConfig::label, f -> f));
        this.root = new FeatureNode<>(features.getFirst(), features, featureCache, predictionHandler.newHandlerInstance(), true);
        this.pruningWindow = pruningWindow;

        updateCleanerTask(pruningWindow);
    }

    /**
     * Construct a FuzzyBucketTree with the associated features and prediction handler. In most cases,
     * this is the result of an auto-tuning exploration with {@link FuzzyBucketTuner}. This
     * constructor has no pruning enabled and all leafs will be retained forever.
     * @param features An ordered list of {@link FeatureConfig} indicating the features the tree will comprise,
     *                 the per feature value count, and finally the per feature bucket value if applicable.
     * @param predictionHandler A {@link PredictionHandler} instance which is respnsible for returning the
     *                          prediction results from a leaf note. Common ones in {@link com.nimbus.fuzzybuckettree.prediction.PredictionHandlers}
     *                          or users may implement their own custom prediction logic, e.g. if a minimum number of
     *                          samples is required to consider a valid result.
     */
    public FuzzyBucketTree(List<FeatureConfig> features, PredictionHandler<T> predictionHandler) {
        this(features, predictionHandler, Duration.ZERO);
    }

    private void updateCleanerTask(Duration newPruningWindow) {
        if (this.pruningWindow != null && newPruningWindow != null && this.pruningWindow.equals(newPruningWindow)
                && this.cleanerTask != null)
            return; // nothing to do it seems

        this.pruningWindow = newPruningWindow;

        if (this.cleanerTask != null) {
            this.cleanerTask.cancel(false);
            this.cleanerTask = null;
        }

        if (pruningWindow != null && pruningWindow.isPositive()) {
            this.cleanerTask = EXECUTOR.scheduleAtFixedRate(() -> {
                try {
                    int pruned = this.root.prune();
                    System.out.println("Pruned entries: " + pruned);
                } catch (Exception e) {
                    System.out.println("Failed to prune entries: " + e.getMessage());
                    e.printStackTrace();
                }
            }, pruningWindow.toSeconds(), pruningWindow.toSeconds(), TimeUnit.SECONDS);
        }
    }

    private void setPruningWindow(Duration pruningWindow) {
        this.pruningWindow = pruningWindow;
    }

    /**
     * @return The {@link FeatureConfig} configuration for this model, in their respective decisioning order.
     */
    public List<FeatureConfig> getFeatures() {
        return features;
    }

    private FeatureValuePair[] getFeatureValuePairs(Map<String, ? extends Object[]> featureValueMap) {
        if (featureValueMap.size() > features.size())
            throw new IllegalArgumentException("Number of prediction features cannot be larger than the number of configured features");

        FeatureValuePair[] featureValuePairs = new FeatureValuePair[features.size()];

        for (int i = 0; i < featureValuePairs.length; i++) {
            FeatureConfig feature = features.get(i);
            featureValuePairs[i] = new FeatureValuePair(feature.label(),
                    featureValueMap.get(feature.label()),
                            feature.type());
        }

        return featureValuePairs;
    }

    /**
     * Make a prediction!
     * @param featureValueMap A map of the feature name exactly as it matches this tree's feature config,
     *                        to each of that feature's value(s). A value must be present for each feature.
     * @return The prediction result type associated with the registered prediction handler, or a prediction
     * with a null value and 0 confidence if no prediction was able to be made.
     */
    public NodePrediction<T> predict(Map<String, ? extends Object[]> featureValueMap) {
        if (featureValueMap == null)
            throw new IllegalArgumentException("Features cannot be null");
        if (featureValueMap.size() != features.size())
            throw new IllegalArgumentException("Number of prediction features does not match the number of configured features");

        NodePrediction<T> p = root.predict(getFeatureValuePairs(featureValueMap));

        return p == null ? NO_PRED : p;
    }

    /**
     * Update this model with one entry of new feature/value information.
     * @param featureValueMap The feature -> value(s) map
     * @param outcome Associated target prediction answer
     */
    public void train(Map<String, ? extends Object[]> featureValueMap, T outcome) {
        if (outcome == null)
            throw new IllegalArgumentException("Prediction outcome cannot be null");

        root.update(getFeatureValuePairs(featureValueMap), outcome);
    }

    public List<FeatureNode<?, T>> nodeTrace(Map<String, float[]> featureValueMap) {
        throw new UnsupportedOperationException("getNodeTrace not impl yet");
    }

    /**
     * Save model data to the provided file path for reloading later
     * @param path File path to save to, e.g. /data/mymodel.fbt
     * @throws IOException
     */
    public void saveModel(String path) throws IOException {
        try (OutputStream outputStream = Files.newOutputStream(Paths.get(path))) {
            ModelObjectMapper.MAPPER.writeValue(outputStream, this);
        } catch (IOException e) {
            throw new IOException("Failed to save model to " + path, e);
        }
    }

    /**
     * Merges another FuzzyBucketTree into this one.
     * Both trees must have identical feature configurations and prediction handler types.
     * @param other The tree to merge into this one
     */
    public void merge(FuzzyBucketTree<T> other) {
        if (other == null) {
            throw new IllegalArgumentException("Cannot merge with null tree");
        }

        // Basic validation of feature compatibility
        if (!features.equals(other.features)) {
            throw new IllegalArgumentException("Cannot merge trees with different feature configurations");
        }

        root.merge(other.root);
    }

    /**
     * @return Number of unique leaf counts in the tree
     */
    public int leafCount() {
        return this.root.leafCount();
    }

    /**
     * @return Interval at which nodes/leafs are evaluated for pruning
     */
    public Duration getPruningWindow() {
        return this.pruningWindow;
    }

    public void enableOrUpdatePruning(Duration pruningWindow) {
        updateCleanerTask(pruningWindow);
    }

    /**
     * Shutdown cleaning tasks and perform any other cleanup work
     * if this tree instance will no longer be used. Calls
     * {@link FeatureNode#shutdown()} for every child node,
     * which in turn calls {@link PredictionHandler#cleanup()}.
     * Should not modify data, but should perform any tasks
     * cleanup.
     */
    public void shutdown() {
        this.cleanerTask.cancel(true);
        this.root.shutdown();
    }

}
