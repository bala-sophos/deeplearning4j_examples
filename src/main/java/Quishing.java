import com.google.zxing.BinaryBitmap;
import com.google.zxing.LuminanceSource;
import com.google.zxing.NotFoundException;
import com.google.zxing.Result;
import com.google.zxing.client.j2se.BufferedImageLuminanceSource;
import com.google.zxing.common.HybridBinarizer;
import com.google.zxing.qrcode.QRCodeReader;
import jakarta.mail.Multipart;
import jakarta.mail.Part;
import jakarta.mail.internet.MimeMessage;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public class Quishing
{
    private static String samplesDir = "/Users/b.rajkumar/Downloads/corpus";
    private static ComputationGraph model;

    static
    {
        try
        {
            model = ModelSerializer.restoreComputationGraph(new File("trained_qr_model_vgg16_6.zip"));
        }
        catch (IOException e)
        {
            e.printStackTrace();
        }
    }

    public static void checkSamples(String samplesDir)
    {

    }
    public static void main(String[] args) throws IOException
    {
        Path path = Paths.get(samplesDir);

        Files.walk(path).forEach(file -> {

            try
            {
                if (file.toString().equalsIgnoreCase(samplesDir))
                {
                    return;
                }

                System.out.println("=====================");
                System.out.println("Checking " + file.toString());
                MimeMessage mimeMessage = new MimeMessage(
                    null,
                    new FileInputStream(file.toString())
                );

                extractImageAttachments(file.getFileName().toString(), mimeMessage);

            }
            catch(Exception e)
            {
                System.out.println("Error processing " + file + " " + e);
            }
        });


    }



    private static String getTextFromQRCode(InputStream is)
        throws Exception
    {
        // Create a BufferedImage from the byte array
        BufferedImage image = ImageIO.read(is);

        // Process the image using ZXing
        LuminanceSource source = new BufferedImageLuminanceSource(image);
        BinaryBitmap bitmap = new BinaryBitmap(new HybridBinarizer(source));
        QRCodeReader reader = new QRCodeReader();


        Result result = reader.decode(bitmap);
        String encodedText = result.getText();
        return encodedText;

    }

    private static boolean isItQRCode(InputStream imageFile)
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
            return true;
        }
        else {
            return false;
        }
    }


    private static void extractImageAttachments(String fileName, Part part) throws Exception {
        if (part.isMimeType("multipart/*")) {
            Multipart multiPart = (Multipart) part.getContent();
            for (int i = 0; i < multiPart.getCount(); i++) {
                extractImageAttachments(fileName, multiPart.getBodyPart(i));
            }
        } else if (part.isMimeType("image/*")) {
            String imageFileName = part.getFileName();
            InputStream is = part.getInputStream();
            long start = System.nanoTime();
            boolean containsQRCode = isItQRCode(part.getInputStream());
            long finishTimeForQRCode = System.nanoTime();
            String qrCodeText = "";
            if (containsQRCode)
            {
                try
                {
                    qrCodeText = getTextFromQRCode(part.getInputStream());
                }
                catch(NotFoundException e)
                {

                }


            }

            long finish = System.nanoTime();
            System.out.println("File: " + imageFileName + "Contains QR Code: " + containsQRCode);
            System.out.println("QR Code text" + qrCodeText);
            System.out.println("Time For Checking QR Code:" + (finishTimeForQRCode - start));
            System.out.println("Time For Reading QR Code:" + (finish - finishTimeForQRCode));
            is.close();

        }
    }
}
