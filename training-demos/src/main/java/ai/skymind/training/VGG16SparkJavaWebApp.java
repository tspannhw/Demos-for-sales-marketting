package ai.skymind.training;

/**
 * Created by tomhanlon on 7/14/17.
 */
import org.nd4j.linalg.factory.Nd4j;
import org.deeplearning4j.nn.modelimport.keras.trainedmodels.Utils.ImageNetLabels;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import javax.servlet.MultipartConfigElement;
import java.io.File;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import static spark.Spark.options;
import static spark.Spark.get;
import static spark.Spark.post;
import static spark.Spark.staticFiles;

import org.nd4j.shade.jackson.databind.ObjectMapper;

/**
 * Created by tomhanlon on 1/25/17.
 */
public class VGG16SparkJavaWebApp {
    public static void main(String[] args) throws Exception {

        File locationToSave = new File("/tmp/vgg16.zip");
        ComputationGraph vgg16 = ModelSerializer.restoreComputationGraph(locationToSave);


        // make upload directory
        File uploadDir = new File("upload");
        uploadDir.mkdir(); // create the upload directory if it doesn't exist

        // form
        String form = "<form method='post' action='getPredictions' enctype='multipart/form-data'>\n" +
                "    <input type='file' name='uploaded_file'>\n" +
                "    <button>Upload picture</button>\n" +
                "</form>";

        staticFiles.location("/Users/tomhanlon/SkyMind/webcontent");// Static files
        //CorsFilter.apply();
        //options("/", (req, res) -> {
        //Appease something
        //   });
        options("/*", (req, res) -> "Hello World");
        get("/hello", (req, res) -> "Hello World");
        get("predict", (req, res) -> form);
        //post("getPredictions",(req, res) -> "GET RESULTS");

        post("/getPredictions", (req, res) -> {

            Path tempFile = Files.createTempFile(uploadDir.toPath(), "", "");

            req.attribute("org.eclipse.jetty.multipartConfig", new MultipartConfigElement("/temp"));

            try (InputStream input = req.raw().getPart("uploaded_file").getInputStream()) { // getPart needs to use same "name" as input field in form
                Files.copy(input, tempFile, StandardCopyOption.REPLACE_EXISTING);
            }

            //logInfo(req, tempFile);
            //return "<h1>You uploaded this image:<h1><img src='" + tempFile.getFileName() + "'>";

            long pipelineTime = System.currentTimeMillis();

            File file = tempFile.toFile();
            //File file = new File(path);
            NativeImageLoader loader = new NativeImageLoader(224, 224, 3);
            INDArray image = loader.asMatrix(file);
            file.delete();
            DataNormalization scaler = new VGG16ImagePreProcessor();

            scaler.transform(image);
            //System.out.print(image);

            long ffTime = System.currentTimeMillis();
            INDArray[] output = vgg16.output(false,image);
            //ffTime = ffTime - System.currentTimeMillis();
            ffTime = System.currentTimeMillis() - ffTime;

            // sort to get top 5
            INDArray[] sorted = Nd4j.sortWithIndices(output[0], 1, false);
            // sorted map for results
            //Map<Float, String> map = new TreeMap<Float, String>(Collections.reverseOrder());
            //VGGResults vggResults = new VGGResults(label,pred);
            ImageNetResults[] vggResultsArray  = new ImageNetResults[5];

            // finish benchmark
            //pipelineTime = pipelineTime - System.currentTimeMillis();
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
            // ##### I AM HERE #####
            ObjectMapper mapper = new ObjectMapper();
            String predictions = mapper.writeValueAsString(vggResultsArray);
            //String predictions = mapper.writeValueAsString(map);
            String predictionmunge = "{" +
                    "\"data\":" + predictions +
                    ", \"performance\":{ \"feedforward\":" + ffTime + ",\"total\":" + pipelineTime + "}" +
                    ", \"network\":{ \"parameters\":" + vgg16.numParams() + ",\"layers\":" + vgg16.getNumLayers() + "}}";
            // return "<h4> '" + predictions  + "' </h4>" +
            //        "Would you like to try another" +
            //        form;
            return predictionmunge ;
            //return "<h1>Your image is: '" + tempFile.getName(1).toString() + "' </h1>";

			      /*
			// Jackson obect mapper
			// ##### I AM HERE #####
			ObjectMapper mapper = new ObjectMapper();
			String predictions = mapper.writeValueAsString(vggResultsArray);
			//String predictions = mapper.writeValueAsString(map);
			            String predictionmunge = "{" +
					                    "\"data\":" + predictions +
					", \"performance\":{ \"feedforward\":" + ffTime + ",\"total\":" + pipelineTime + "}}";
				    // return "<h4> '" + predictions  + "' </h4>" +
				    //        "Would you like to try another" +
				    //        form;
				    return predictionmunge ;
				    //return "<h1>Your image is: '" + tempFile.getName(1).toString() + "' </h1>";
				    */
        });


    }

}

