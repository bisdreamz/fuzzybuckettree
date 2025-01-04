package com.nimbus.fuzzybuckettree;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.nimbus.fuzzybuckettree.prediction.NodePrediction;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.concurrent.ConcurrentHashMap;

public class MultiTree<T> {

    @JsonProperty
    private final Map<String, FuzzyBucketTree<T>> domains;

    public MultiTree() {
        this.domains = new ConcurrentHashMap<>();
    }

    public void put(String domain, FuzzyBucketTree<T> tree) {
        if (domain == null)
            throw new NullPointerException("domain is null");
        if (tree == null)
            throw new NullPointerException("tree is null");
        if (domains.containsKey(domain))
            throw new IllegalArgumentException("domain already exists. use replace() to tree.merge()");

        this.domains.put(domain, tree);
    }

    public void replace(String domain, FuzzyBucketTree<T> tree) {
        if (domain == null)
            throw new NullPointerException("domain is null");
        if (tree == null)
            throw new NullPointerException("tree is null");
        if (!domains.containsKey(domain))
            throw new IllegalArgumentException("domain to replace doesnt exist");

        this.domains.put(domain, tree);
    }

    public NodePrediction predict(String domain, Map<String, float[]> featureValueMap) {
        if (domain == null)
            throw new NullPointerException("domain is null");
        if (featureValueMap == null || featureValueMap.isEmpty())
            throw new NullPointerException("features is null or empty");

        FuzzyBucketTree<T> tree = this.domains.get(domain);
        if (tree == null)
            throw new NoSuchElementException("No configured prediction domain exists for domain " + domain);

        return tree.predict(featureValueMap);
    }

    /**
     * Update this model with one entry of new feature/value information for the provided domain.
     * @param domain The domain this update applies to
     * @param featureValueMap The feature -> value(s) map
     * @param outcome Associated target prediction answer
     */
    public void train(String domain, Map<String, float[]> featureValueMap, T outcome) {
        if (domain == null)
            throw new NullPointerException("domain is null");

        FuzzyBucketTree<T> tree = this.domains.get(domain);
        if (tree == null)
            throw new NoSuchElementException("No configured prediction domain exists for domain " + domain);

        tree.train(featureValueMap, outcome);
    }

    /**
     * Save model data to the provided file path for reloading later
     * @param path File path to save to, e.g. /data/mymodel.mt
     * @throws IOException if there's an error saving the model
     */
    public void saveModel(String path) throws IOException {
        ObjectMapper mapper = new ObjectMapper();

        try (OutputStream outputStream = Files.newOutputStream(Paths.get(path))) {
            mapper.writeValue(outputStream, this);
        } catch (IOException e) {
            throw new IOException("Failed to save model to " + path, e);
        }
    }

    /**
     * Load a MultiTree model from the specified path
     * @param path Path to load the model from
     * @return The loaded MultiTree model
     * @throws IOException if there's an error loading the model
     */
    public static <T> MultiTree<T> loadModel(String path) throws IOException {
        try (InputStream inputStream = Files.newInputStream(Paths.get(path))) {
            return ModelObjectMapper.MAPPER.readValue(inputStream,
                    ModelObjectMapper.MAPPER.getTypeFactory().constructParametricType(MultiTree.class, Object.class));
        } catch (IOException e) {
            throw new IOException("Failed to load model from " + path, e);
        }
    }

    /**
     * Merges another MultiTree into this one
     * @param other The MultiTree to merge from
     */
    public void merge(MultiTree<T> other) {
        if (other == null) {
            throw new NullPointerException("Cannot merge null MultiTree");
        }

        other.domains.forEach((domain, otherTree) -> {
            FuzzyBucketTree<T> thisTree = this.domains.get(domain);
            if (thisTree == null)
                this.domains.put(domain, otherTree);
            else
                thisTree.merge(otherTree);
        });
    }

}
