import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Nadam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.Random;

//Based on MNISTDoubleLayer
public class TrainQRCodeClassifier2
{
    private static       Logger    log               = LoggerFactory.getLogger(TrainQRCodeClassifier2.class);
    private static final String [] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;
    private static final long seed = 12345;

    private static final Random randNumGen = new Random(seed);

    public static void main(String[] args)
        throws IOException
    {
        // Image dimensions
        int height = 100;
        int width = 100;
        int channels = 3;
        int rngSeed = 123; // random number seed for reproducibility
        double rate = 0.0015; // learning rate

        //DIRECTORY STRUCTURE:
        //Images in the dataset have to be organized in directories by class/label.
        //In this example there are ten images in three classes
        //Here is the directory structure
        //                                    parentDir
        //                                  /    |     \
        //                                 /     |      \
        //                            labelA  labelB   labelC
        //
        //Set your data up like this so that labels from each label/class live in their own directory
        //And these label/class directories live together in the parent directory
        //
        //
        File parentDir=new File("/Users/b.rajkumar/Downloads/quishing/training");
        //Files in directories under the parent dir that have "allowed extensions" split needs a random number generator for reproducibility when splitting the files into train and test
        FileSplit filesInDir = new FileSplit(parentDir, allowedExtensions, randNumGen);

        //You do not have to manually specify labels. This class (instantiated as below) will
        //parse the parent dir and use the name of the subdirectories as label/class names
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        //The balanced path filter gives you fine tune control of the min/max cases to load for each class
        //Below is a bare bones version. Refer to javadoc for details
        BalancedPathFilter pathFilter = new BalancedPathFilter(randNumGen, allowedExtensions, labelMaker);

        //Split the image files into train and test. Specify the train test split as 80%,20%
        InputSplit[] filesInDirSplit = filesInDir.sample(pathFilter, 80, 20);
        InputSplit trainData = filesInDirSplit[0];
        InputSplit testData = filesInDirSplit[1];

        //Specifying a new record reader with the height and width you want the images to be resized to.
        //Note that the images in this example are all of different size
        //They will all be resized to the height and width specified below
        ImageRecordReader recordReader = new ImageRecordReader(height,width,channels,labelMaker);
        ImageRecordReader testRecordReader = new ImageRecordReader(height,width,channels,labelMaker);

        //Initialize the record reader with the train data and the transform chain
        recordReader.initialize(trainData);
        testRecordReader.initialize(testData);

        int outputNum = recordReader.numLabels();

        System.out.println("Output labels count:" + outputNum);
        //convert the record reader to an iterator for training - Refer to other examples for how to use an iterator
        int batchSize = 10; // Minibatch size. Here: The number of images to fetch for each call to dataIter.next().
        int labelIndex = 1; // Index of the label Writable (usually an IntWritable), as obtained by recordReader.next()


        DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, labelIndex, outputNum);
        DataSetIterator testDataIter = new RecordReaderDataSetIterator(testRecordReader, batchSize, labelIndex, outputNum);

        ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);
        dataIter.setPreProcessor(scaler);
        testDataIter.setPreProcessor(scaler);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(rngSeed) //include a random seed for reproducibility
            .activation(Activation.RELU)
            .weightInit(WeightInit.XAVIER)
            .updater(new Nadam())
            .l2(rate * 0.005) // regularize learning model
            .list()
            .layer(new DenseLayer.Builder() //create the first input layer.
                                            .nIn(height * width * channels)
                                            .nOut(1500)
                                            .build())
            .layer(new DenseLayer.Builder() //create the second input layer
                                            .nIn(1500)
                                            .nOut(500)
                                            .build())
            .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD) //create hidden layer
                                                                                             .activation(Activation.SOFTMAX)
                                                                                             .nOut(outputNum)
                                                                                             .build())
            .inputType().convolutional(height, width, channels)
            .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(1));

        log.info("Train model....");
        model.fit(dataIter, 20);

        log.info("Evaluate model....");
        Evaluation eval = model.evaluate(testDataIter);

        log.info(eval.stats());


        // Save the trained model
        ModelSerializer.writeModel(model, new File("trained_qr_model.zip"), true);
    }
}
