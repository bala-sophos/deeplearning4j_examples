import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.Random;

//Based on VGG16 pre-trained model
public class TrainQRCodeClassifierVGG16
{
    private static       Logger    log               = LoggerFactory.getLogger(TrainQRCodeClassifierVGG16.class);
    private static final String [] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;
    private static final long seed = 12345;

    private static final Random randNumGen = new Random(seed);

    // Image dimensions
    private static int height = 224;
    private static int width = 224;
    private static int channels = 3;

    // learning rate
    private static double rate = 0.0015;

    private static ImageRecordReader recordReader = null;
    private static ImageRecordReader testRecordReader = null;

    private static DataSetIterator dataIter = null;
    private static DataSetIterator testDataIter = null;

    private static int numLabels;

    private static void initialiseDataReaders(final String srcDir)
        throws IOException
    {
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
        //File parentDir=new File("/Users/b.rajkumar/Downloads/quishing/training");
        File parentDir=new File(srcDir);
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
        recordReader = new ImageRecordReader(height,width,channels,labelMaker);
        testRecordReader = new ImageRecordReader(height,width,channels,labelMaker);


        //Initialize the record reader with the train data and the transform chain
        recordReader.initialize(trainData);

        testRecordReader.initialize(testData);


        numLabels = recordReader.numLabels();

        System.out.println("Output labels count:" + numLabels);

        //convert the record reader to an iterator for training - Refer to other examples for how to use an iterator
        int batchSize = 16; // Minibatch size. Here: The number of images to fetch for each call to dataIter.next().
        int labelIndex = 1; // Index of the label Writable (usually an IntWritable), as obtained by recordReader.next()


        dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, labelIndex, numLabels);
        testDataIter = new RecordReaderDataSetIterator(testRecordReader, batchSize, labelIndex, numLabels);

        ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);
        dataIter.setPreProcessor(scaler);
        testDataIter.setPreProcessor(scaler);

    }

    public static void main(String[] args)
        throws IOException
    {

        //https://deeplearning4j.konduit.ai/v/en-1.0.0-beta7/tuning-and-training/transfer-learning
        //https://deeplearning4j.konduit.ai/v/en-1.0.0-beta7/model-zoo/overview

        initialiseDataReaders("/Users/b.rajkumar/Downloads/quishing/qr_code_knife_pistol_random/training");

        ZooModel zooModel = VGG16.builder().build();
        ComputationGraph pretrainedNet = (ComputationGraph) zooModel.initPretrained(PretrainedType.IMAGENET);


        FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .updater(new Nesterovs(5e-5))
            .seed(seed)
            .build();

        ComputationGraph vgg16Transfer = new TransferLearning.GraphBuilder(pretrainedNet)
            .fineTuneConfiguration(fineTuneConf)
            .setFeatureExtractor("fc2")
            .removeVertexKeepConnections("predictions")
            .addLayer("predictions",
                      new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                          .nIn(4096).nOut(numLabels)
                          .weightInit(WeightInit.XAVIER)
                          .activation(Activation.SOFTMAX).build(), "fc2")
            .build();

        vgg16Transfer.setListeners(new ScoreIterationListener(10));


        log.info("Train model....");
        vgg16Transfer.fit(dataIter, 10);

        log.info("Evaluate model....");
        Evaluation eval = vgg16Transfer.evaluate(testDataIter);

        log.info(eval.stats());


        // Save the trained model
        ModelSerializer.writeModel(vgg16Transfer, new File("trained_qr_model_vgg16_6.zip"), true);
    }
}

//trained_qr_model_vgg16  - Using an existing model VGG16
//trained_qr_model_vgg16_2.zip - with plain email samples in not qr code. - accuracy was very low
//trained_qr_model_vgg16_3.zip - Using three output classes.
//trained_qr_model_vgg16_4.zip - Going back. Just 2 classes. Not using text only images under 'not_qr_code' /Users/b.rajkumar/Downloads/quishing/training
//trained_qr_model_vgg16_5.zip - training_data_with_qr_code_only_and_random
//trained_qr_model_vgg16_6.zip - /Users/b.rajkumar/notebooks/training_qr_code_knife_pistol. Four labels knife, pistol, qr_code, random.
