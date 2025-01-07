package com.nimbus.fuzzybuckettree;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonTypeInfo;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.nimbus.fuzzybuckettree.prediction.NodePrediction;
import com.nimbus.fuzzybuckettree.prediction.Prediction;
import com.nimbus.fuzzybuckettree.prediction.handlers.PredictionHandler;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class FuzzyBucketTree<T> {

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

    @JsonCreator
    public FuzzyBucketTree(
            @JsonProperty("root") FeatureNode<?, T> root,
            @JsonProperty("features") List<FeatureConfig> features) {
        this.root = root;
        this.features = features;
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
     */
    public FuzzyBucketTree(List<FeatureConfig> features, PredictionHandler<T> predictionHandler) {
        this.features = features;

        if (features == null || features.isEmpty())
            throw new IllegalArgumentException("Features cannot be empty");


        Map<String, FeatureConfig> featureCache = features.stream().collect(Collectors.toMap(FeatureConfig::label, f -> f));
        this.root = new FeatureNode<>(features.getFirst(), features, featureCache, predictionHandler.newHandlerInstance(), true);
    }

    private FeatureValuePair[] getFeatureValuePairs(Map<String, ? extends Object[]> featureValueMap) {
        if (featureValueMap.size() > features.size())
            throw new IllegalArgumentException("Number of prediction features cannot be larger than the number of configured features");

        FeatureValuePair[] featureValuePairs = new FeatureValuePair[features.size()];

        for (int i = 0; i < featureValuePairs.length; i++) {
            FeatureConfig feature = features.get(i);
            featureValuePairs[i] = new FeatureValuePair(feature.label(), featureValueMap.get(feature.label()));
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
     * Merges another FuzzyBucketTree into this one, combining their prediction data.
     * Both trees must have identical feature configurations and prediction handler types.
     *
     * @param other The tree to merge into this one
     * @throws IllegalArgumentException if the trees are incompatible
     */
    /**
     * Merges another FuzzyBucketTree into this one, combining their prediction data.
     * Both trees must have identical feature configurations and prediction handler types.
     *
     * @param other The tree to merge into this one
     * @throws IllegalArgumentException if the trees are incompatible
     */
    /**
     * Merges another FuzzyBucketTree into this one.
     * Both trees must have identical feature configurations and prediction handler types.
     * @param other The tree to merge into this one
     */
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

}
