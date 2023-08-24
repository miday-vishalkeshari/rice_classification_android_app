package com.example.a27062023evening

import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.os.Bundle
import com.example.a27062023evening.R
import android.provider.MediaStore
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.FileInputStream
import java.io.IOException
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import android.widget.Toast

class MainActivity : AppCompatActivity() {

    private lateinit var tfliteInterpreter: Interpreter
    private lateinit var imageView: ImageView
    private lateinit var predictionTextView: TextView
    private lateinit var selectedImageBitmap: Bitmap
    private var classLabels: List<String> = emptyList()
    private lateinit var scrollTextView: TextView
    private val SELECT_IMAGE_REQUEST = 1
    private val CAPTURE_IMAGE_REQUEST = 2

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        imageView = findViewById(R.id.imageView)
        predictionTextView = findViewById(R.id.predictionTextView)
        scrollTextView = findViewById(R.id.scroll_view_text)
        val selectGalleryButton: Button = findViewById(R.id.selectGalleryButton)
        val selectCameraButton: Button = findViewById(R.id.captureCameraButton)

        // Load the TFLite model
        loadModel()

        selectGalleryButton.setOnClickListener {
            // Open the gallery to select an image
            val intent = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
            startActivityForResult(intent, SELECT_IMAGE_REQUEST)
        }

        selectCameraButton.setOnClickListener {
            // Open the camera to capture an image
            val intent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
            startActivityForResult(intent, CAPTURE_IMAGE_REQUEST)
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == SELECT_IMAGE_REQUEST && resultCode == Activity.RESULT_OK && data != null) {
            // Get the selected image URI
            val imageUri = data.data
            try {
                // Load the selected image bitmap
                val inputStream = contentResolver.openInputStream(imageUri!!)
                selectedImageBitmap = BitmapFactory.decodeStream(inputStream)
                // Set the bitmap to the ImageView
                imageView.setImageBitmap(selectedImageBitmap)
                // Perform prediction on the selected image
                performPrediction()
            } catch (e: IOException) {
                e.printStackTrace()
            }
        } else if (requestCode == CAPTURE_IMAGE_REQUEST && resultCode == Activity.RESULT_OK && data != null) {
            // Get the captured image bitmap
            selectedImageBitmap = data.extras?.get("data") as Bitmap
            // Set the bitmap to the ImageView
            imageView.setImageBitmap(selectedImageBitmap)
            // Perform prediction on the captured image
            performPrediction()
        }
    }

    private fun loadModel() {
        try {
            val modelFileDescriptor = assets.openFd("model.tflite")
            val inputStream = FileInputStream(modelFileDescriptor.fileDescriptor)
            val modelFile = inputStream.channel.map(
                FileChannel.MapMode.READ_ONLY,
                modelFileDescriptor.startOffset,
                modelFileDescriptor.declaredLength
            )
            val interpreterOptions = Interpreter.Options()
            tfliteInterpreter = Interpreter(modelFile, interpreterOptions)

            // Load the class labels from the file
            val labelsInput = assets.open("labels.txt")
            val labelsBytes = ByteArray(labelsInput.available())
            labelsInput.read(labelsBytes)
            labelsInput.close()

            // Convert the byte array to a string and split into labels
            val labelsString = String(labelsBytes, Charsets.UTF_8)
            classLabels = labelsString.trim().split("\n")
        } catch (e: IOException) {
            e.printStackTrace()
        }
    }

    private fun performPrediction() {
        // Resize the selected image to match the required dimensions
        val resizedImage = Bitmap.createScaledBitmap(
            selectedImageBitmap,
            256,
            256,
            true
        )

        // Preprocess the normalized image
        val normalizedImage = normalizeImage(resizedImage)

        // Preprocess the resized image
        val inputImage = TensorImage(DataType.FLOAT32)
        inputImage.load(normalizedImage)
        val imageBuffer = TensorBuffer.createFixedSize(
            intArrayOf(1, 256, 256, 3),
            DataType.FLOAT32
        )
        imageBuffer.loadBuffer(inputImage.buffer)

        // Perform inference
        val outputShape = tfliteInterpreter.getOutputTensor(0).shape()
        if (outputShape.contentEquals(intArrayOf(1, 5))) {
            val outputBuffer = TensorBuffer.createFixedSize(intArrayOf(1, 5), DataType.FLOAT32)
            tfliteInterpreter.run(imageBuffer.buffer, outputBuffer.buffer.rewind())

            // Get the predicted class
            val prediction = outputBuffer.floatArray

            if (prediction.isNotEmpty()) {
                val predictedIndex = prediction.indices.maxByOrNull { prediction[it] } ?: -1

                if (predictedIndex >= 0 && predictedIndex < classLabels.size) {
                    val predictedClass = classLabels[predictedIndex]
                    Toast.makeText(applicationContext, "score= " + prediction[predictedIndex], Toast.LENGTH_LONG).show()

                    // Check if prediction score is less than 50%
                    if (prediction[predictedIndex] < 0.5) {
                        // Show toast message
                        Toast.makeText(applicationContext, "Can't able to predict", Toast.LENGTH_LONG).show()

                        // Clear the scrollTextView
                        scrollTextView.text = "Can't able to predict anything. Please upload a proper image."
                    } else {
                        Toast.makeText(applicationContext, "Hey, we found", Toast.LENGTH_LONG).show()

                        // Display the prediction in the scrollTextView
                        val classText = classTextData[predictedIndex]
                        if (classText != null) {
                            scrollTextView.text = "Predicted Class: $predictedClass\n\n$classText"
                        } else {
                            scrollTextView.text = "Predicted Class: $predictedClass"
                        }
                    }
                } else {
                    // Handle invalid predicted index
                    scrollTextView.text = "Unknown prediction"
                }
            } else {
                // Handle empty prediction buffer
                scrollTextView.text = "Prediction error"
            }
        } else {
            // Handle incorrect output shape
            scrollTextView.text = "Incorrect output shape"
        }
    }









    private fun normalizeImage(bitmap: Bitmap): Bitmap {
        val normalizedImage = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        for (y in 0 until normalizedImage.height) {
            for (x in 0 until normalizedImage.width) {
                val pixel = bitmap.getPixel(x, y)
                val red = Color.red(pixel) / 255.0
                val green = Color.green(pixel) / 255.0
                val blue = Color.blue(pixel) / 255.0
                val normalizedPixel = Color.argb(255, (red * 255).toInt(), (green * 255).toInt(), (blue * 255).toInt())
                normalizedImage.setPixel(x, y, normalizedPixel)
            }
        }
        return normalizedImage
    }


    companion object {
        private val classTextData: Map<Int, String> = mapOf(
            0 to "Arborio rice is a type of short-grain rice known for its starchy and creamy texture. \n\nIt is primarily grown in the Piedmont region of Italy. \n" +
                    "\nArborio rice is commonly used in risotto, a traditional Italian dish. \n" +
                    "\nIt has a high starch content, which gives risotto its characteristic creamy consistency. \n" +
                    "\nArborio rice has a round shape and a pearly white color. \n" +
                    "\nIt absorbs flavors well, making it suitable for various risotto recipes. \n" +
                    "\nThe grains of Arborio rice are firm and chewy, with a slightly sticky texture. \n" +
                    "\nIt requires regular stirring during cooking to release its starch and create a creamy texture. \n" +
                    "\nArborio rice is versatile and can be used in other dishes like rice pudding or rice salads. \n" +
                    "\nIt is widely available in grocery stores and specialty food markets.",
            1 to "Basmati rice is a long-grain rice known for its distinct aroma and flavor. \n" +
                    "\nIt is primarily grown in the Indian subcontinent. \n" +
                    "\nBasmati rice is prized for its fluffy texture and delicate, nutty taste. \n" +
                    "\nIt is commonly used in various Indian, Middle Eastern, and Asian dishes. \n" +
                    "\nBasmati rice has a slender grain and a light beige color. \n" +
                    "\nIt is often used to make biryani, pilaf, and other rice-based dishes. \n" +
                    "\nThe grains of Basmati rice elongate and separate when cooked properly. \n" +
                    "\nIt is known for its fragrance, often described as a blend of flowers and nuts. \n" +
                    "\nBasmati rice is popular for its ability to absorb flavors and spices. \n" +
                    "\nIt is considered a premium variety of rice and is widely sought after.",
            2 to "Ipsala rice is a medium-grain rice cultivated in Ipsala, a region in Turkey. \n" +
                    "\nIt is known for its tender texture and slightly sweet flavor. \n" +
                    "\nIpsala rice is commonly used in pilaf, rice pudding, and other Turkish dishes. \n" +
                    "\nIt has a slightly sticky consistency when cooked. \n" +
                    "\nIpsala rice is often preferred for its ability to absorb flavors and spices. \n" +
                    "\nThe grains of Ipsala rice are plump and moist. \n" +
                    "\nIt is a popular choice for both savory and sweet rice dishes in Turkish cuisine. \n" +
                    "\nIpsala rice is recognized for its high-quality and is favored by chefs. \n" +
                    "\nIt is available in both white and brown varieties. \n" +
                    "\nIpsala rice adds a delightful taste and texture to a wide range of recipes.",
            3 to "Jasmine rice is an aromatic long-grain rice commonly used in Southeast Asian cuisine. \n" +
                    "\nIt is native to Thailand and is known for its fragrant aroma and delicate flavor. \n" +
                    "\nJasmine rice has a soft and slightly sticky texture when cooked. \n" +
                    "\nIt is often used as a side dish or as a base for various Thai and Asian dishes. \n" +
                    "\nJasmine rice has a subtle floral aroma, similar to the scent of jasmine flowers. \n" +
                    "\nThe grains of Jasmine rice are slightly shorter and rounder than other long-grain rice varieties. \n" +
                    "\nIt pairs well with curries, stir-fries, and other flavorful dishes. \n" +
                    "\nJasmine rice is a staple ingredient in Thai cuisine and is enjoyed worldwide. \n" +
                    "\nIt is prized for its versatility, fragrance, and ability to absorb flavors. \n" +
                    "\nJasmine rice is widely available and is a popular choice for many rice-based recipes.",
            4 to "Karacadag rice is a traditional Turkish rice variety named after the Karacadag region. \n" +
                    "\nIt is known for its distinct taste, texture, and aroma. \n" +
                    "\nKaracadag rice is often used in pilaf, meat dishes, and various Turkish rice recipes. \n" +
                    "\nIt has a unique nutty flavor and a tender yet slightly firm texture. \n" +
                    "\nThe grains of Karacadag rice are medium-sized and have a light golden color. \n" +
                    "\nIt is highly regarded for its quality and is favored by rice enthusiasts. \n" +
                    "\nKaracadag rice retains its shape and does not stick together when cooked. \n" +
                    "\nIt adds a rich and savory element to dishes and complements a wide range of flavors. \n" +
                    "\nKaracadag rice is an important part of Turkish culinary heritage. \n" +
                    "\nIt is cherished for its taste, aroma, and its role in creating authentic Turkish meals."
            // Add more class-text data pairs as needed
        )
    }
}

