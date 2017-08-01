package ai.skymind.training;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.trainedmodels.Utils.ImageNetLabels;
import org.deeplearning4j.parallelism.ParallelInference;
import org.deeplearning4j.parallelism.inference.InferenceMode;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.shade.jackson.databind.ObjectMapper;

import javax.servlet.MultipartConfigElement;
import java.io.File;
import java.io.InputStream;

import static spark.Spark.*;

/**
 * Created by tomhanlon on 7/15/17.
 */
public class InceptionWebApp {
    private static final ObjectMapper mapper = new ObjectMapper();
    private static final int imgWidth = 299;
    private static final int imgHeight = 299;
    private static final int imgChannels = 3;
    private static final int numClasses = 1000;
    private static final NativeImageLoader imageLoader = new NativeImageLoader(imgHeight, imgWidth, imgChannels);

    // form to request upload
    private static final String form = "<form method='post' action='getPredictions' enctype='multipart/form-data'>\n" +
            "    <input type='file' name='uploaded_file'>\n" +
            "    <button>Upload picture</button>\n" +
            "</form>";

    public static void main(String[] args) throws Exception {
        File locationToSave = new File("trained_inception_model.zip");
        ComputationGraph model = ModelSerializer.restoreComputationGraph(locationToSave);

        ParallelInference modelWrapper = new ParallelInference.Builder(model)
                .inferenceMode(InferenceMode.BATCHED)
                .batchLimit(5)
                .workers(3)
                .build();

        // spark java webapp stuff
        // urls to respond to
        options("/*", (req, res) -> "Hello World");
        get("/hello", (req, res) -> "Hello World");
        get("predict", (req, res) -> form);

        // generate response
        post("/getPredictions", (req, res) -> {
            req.attribute("org.eclipse.jetty.multipartConfig", new MultipartConfigElement("/temp"));

            long pipelineTime = System.currentTimeMillis();

            INDArray image;
            try (InputStream input = req.raw().getPart("uploaded_file").getInputStream()) { // getPart needs to use same "name" as input field in form
                image = imageLoader.asMatrix(input).divi(255.0).subi(0.5).muli(2);
            }

            long ffTime = System.currentTimeMillis();

            INDArray[] output = modelWrapper.output(new INDArray[]{ image });

            ffTime = System.currentTimeMillis() - ffTime;

            // sort to get top 5
            INDArray[] sorted = Nd4j.sortWithIndices(output[0], 1, false);

            // VGGResults class just builds an array of results in nice format
            ImageNetResults[] vggResultsArray = new ImageNetResults[5];

            // finish benchmark
            pipelineTime = System.currentTimeMillis() - pipelineTime;

            // Get top 5
            for (int i = 0; i < 5; i++) {
                // Get prediction percent
                float prediction = sorted[1].getFloat(i) * 100;

                // extract label for prediction
                String Label = ImageNetLabels.getLabel(sorted[0].getInt(i));

                // put both in Result array
                vggResultsArray[i] = new ImageNetResults(Label, prediction);
            }

            String predictions = mapper.writeValueAsString(vggResultsArray);
            //String predictions = mapper.writeValueAsString(map);
            // output json to screen
            return "{" +
                    "\"data\":" + predictions +
                    ", \"performance\":{ \"feedforward\":" + ffTime + ",\"total\":" + pipelineTime + "}" +
                    ", \"network\":{ \"parameters\":" + model.numParams() + ",\"layers\":" + model.getNumLayers() + "}}";
        });
    }
}
