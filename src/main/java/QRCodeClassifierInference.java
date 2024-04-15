import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.File;
import java.io.IOException;

public class QRCodeClassifierInference {
    public static void main(String[] args) throws IOException {
        // Load the saved model
        //MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(new File("trained_qr_model.zip"));
        ComputationGraph model = ModelSerializer.restoreComputationGraph(new File("trained_qr_model_vgg16_4.zip"));

        String inputFolderPath = "/Users/b.rajkumar/Downloads/quishing/qr_code_augmented/qr_dataset";
        //String inputFolderPath = "/Users/b.rajkumar/Downloads/quishing/qr_code_parcial_labeled/validation_images";

        File inputFolder = new File(inputFolderPath);

        if (!inputFolder.isDirectory() ) {
            System.out.println("Please provide valid input and output folder paths.");
            return;
        }


        try
        {
            // Get list of files in the input folder


            File[] files = inputFolder.listFiles();

            if (files != null)
            {

                /*

                for (File imageFile : files)
                {
                    // Load the image for inference


                    eval(model, imageFile);
                }*/

            }


            long start = System.nanoTime();
            File imageFile = new File("/Users/b.rajkumar/Downloads/quishing/quishing_samples/qr4.png");
            eval(model, imageFile);
            long finish = System.nanoTime();
            long timeElapsed = finish - start;
            System.out.println("Elapsed time:" + timeElapsed);

            imageFile = new File("/Users/b.rajkumar/Downloads/quishing/quishing_samples/qr5.jpg");
            eval(model, imageFile);

            imageFile = new File("/Users/b.rajkumar/Downloads/quishing/quishing_samples/qr6.png");
            eval(model, imageFile);

            imageFile = new File("/Users/b.rajkumar/Downloads/quishing/quishing_samples/qr7.png");
            eval(model, imageFile);

            imageFile = new File("/Users/b.rajkumar/Downloads/quishing/quishing_samples/qr8.png");
            eval(model, imageFile);

            imageFile = new File("/Users/b.rajkumar/Downloads/quishing/quishing_samples/qr9.png");
            eval(model, imageFile);

            imageFile = new File("/Users/b.rajkumar/Downloads/quishing/quishing_samples/qr10.png");
            eval(model, imageFile);

            imageFile = new File("/Users/b.rajkumar/Downloads/quishing/quishing_samples/qr11.png");
            eval(model, imageFile);

            imageFile = new File("/Users/b.rajkumar/Downloads/quishing/quishing_samples/qr-code.png");
            eval(model, imageFile);

            imageFile = new File("/Users/b.rajkumar/Downloads/quishing/quishing_samples/qr4_hru.jpeg");
            eval(model, imageFile);


            imageFile = new File("/Users/b.rajkumar/Downloads/qr_plain_test1.jpeg");
            eval(model, imageFile);


            imageFile = new File("/Users/b.rajkumar/Downloads/apples/apple1.jpg");
            eval(model, imageFile);

            imageFile = new File("/Users/b.rajkumar/Downloads/apples/apple2.jpg");
            eval(model, imageFile);

            imageFile = new File("/Users/b.rajkumar/Downloads/apples/apple3.jpg");
            eval(model, imageFile);

            imageFile = new File("/Users/b.rajkumar/Downloads/apples/plain_text.png");
            eval(model, imageFile);

            imageFile = new File("/Users/b.rajkumar/Downloads/apples/greetings.png");
            eval(model, imageFile);

            imageFile = new File("/Users/b.rajkumar/Downloads/apples/onlygreeting.png");
            eval(model, imageFile);
        }
        catch(Exception e)
        {
            e.printStackTrace();
        }

    }

    private static void eval(ComputationGraph model, File imageFile)
        throws IOException
    {
        NativeImageLoader loader = new NativeImageLoader(224, 224, 3);
        INDArray imageArray = loader.asMatrix(imageFile);

        // Normalize image data
        imageArray.divi(255);



        // Perform inference
        INDArray output = model.outputSingle(imageArray);

        //System.out.println(output.toString());

        // Get prediction
        int predictedClass = output.argMax(1).getInt(0);

        // Print the result
        if (predictedClass == 0) {
            System.out.println(imageFile.getName() + " Random");
        }
        else {
            System.out.println(imageFile.getName() + " QR Code");
        }
    }

    private static void eval(MultiLayerNetwork model, File imageFile)
        throws IOException
    {
        NativeImageLoader loader = new NativeImageLoader(100, 100, 3);
        INDArray imageArray = loader.asMatrix(imageFile);

        // Normalize image data
        imageArray.divi(255);


        // Perform inference
        INDArray output = model.output(imageArray);


        System.out.println(output.toString());

        // Get prediction
        int predictedClass = output.argMax(1).getInt(0);

        if (predictedClass == 0) {
            System.out.println(imageFile.getName() + " Random");
        }
        else {
            System.out.println(imageFile.getName() + " QR Code");
        }
    }
}
