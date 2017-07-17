package ai.skymind.training;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.trainedmodels.Utils.ImageNetLabels;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.shade.jackson.databind.ObjectMapper;

import javax.servlet.MultipartConfigElement;
import java.io.File;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;

import static spark.Spark.*;

/**
 * Created by tomhanlon on 7/15/17.
 */
public class InceptionWebApp {

    public static void main(String[] args) throws Exception {

        int imgWidth = 299;
        int imgHeight = 299;
        int imgChannels = 3;
        int numClasses = 1000;

        File locationToSave = new File("/tmp/trained_inception_model.zip");
        ComputationGraph model = ModelSerializer.restoreComputationGraph(locationToSave);

        // spark java webapp stuff
        // make upload directory for submitted images
        File uploadDir = new File("upload");
        uploadDir.mkdir(); // create the upload directory if it doesn't exist




        // form to request upload
        String form = "<form method='post' action='getPredictions' enctype='multipart/form-data'>\n" +
                "    <input type='file' name='uploaded_file'>\n" +
                "    <button>Upload picture</button>\n" +
                "</form>";

        // urls to respond to
        options("/*", (req, res) -> "Hello World");
        get("/hello", (req, res) -> "Hello World");
        get("predict", (req, res) -> form);

        // generate response
        post("/getPredictions", (req, res) -> {

            Path tempFile = Files.createTempFile(uploadDir.toPath(), "", "");

            req.attribute("org.eclipse.jetty.multipartConfig", new MultipartConfigElement("/temp"));

            try (InputStream input = req.raw().getPart("uploaded_file").getInputStream()) { // getPart needs to use same "name" as input field in form
                Files.copy(input, tempFile, StandardCopyOption.REPLACE_EXISTING);
            }


            long pipelineTime = System.currentTimeMillis();

            File file = tempFile.toFile();


            NativeImageLoader imageLoader = new NativeImageLoader(imgHeight, imgWidth, imgChannels);
            INDArray image = imageLoader.asMatrix(file).div(255.0).sub(0.5).mul(2);
            file.delete();



            long ffTime = System.currentTimeMillis();
            INDArray[] output = model.output(false,image);
            ffTime = System.currentTimeMillis() - ffTime;

            // sort to get top 5
            INDArray[] sorted = Nd4j.sortWithIndices(output[0], 1, false);

            // VGGResults class just builds an array of results in nice format
            ImageNetResults[] vggResultsArray  = new ImageNetResults[5];

            // finish benchmark

            pipelineTime = System.currentTimeMillis() - pipelineTime;


            // Get top 5
            for (int i = 0; i < 5; i++) {

                // Get prediction percent
                Float prediction = sorted[1].getFloat(i) * 100;

                // extract label for prediction
                String Label = ImageNetLabels.getLabel(sorted[0].getInt(i));

                // put both in Result array
                vggResultsArray[i] = new ImageNetResults(Label, prediction);

            }

            // Jackson obect mapper
            // I think this organizes the output into JSON
            ObjectMapper mapper = new ObjectMapper();
            String predictions = mapper.writeValueAsString(vggResultsArray);
            //String predictions = mapper.writeValueAsString(map);
            String predictionmunge = "{" +
                    "\"data\":" + predictions +
                    ", \"performance\":{ \"feedforward\":" + ffTime + ",\"total\":" + pipelineTime + "}" +
                    ", \"network\":{ \"parameters\":" + model.numParams() + ",\"layers\":" + model.getNumLayers() + "}}";
            // output json to screen
            return predictionmunge ;



        });






    }

    }
