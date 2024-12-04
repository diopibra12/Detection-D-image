package com.my.phototest;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.YuvImage;
import android.media.Image;
import android.os.Bundle;
import android.util.Size;
import android.widget.Toast;

import androidx.annotation.OptIn;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.*;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.core.content.ContextCompat;
import androidx.databinding.DataBindingUtil;

import com.google.common.util.concurrent.ListenableFuture;
import com.my.phototest.databinding.CameraXBinding;
import com.my.phototest.ml.MonModele;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;

public class CameraX extends AppCompatActivity {
    CameraXBinding binding;
    ProcessCameraProvider cameraProvider;
    private MonModele model;
    private Executor executor;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = DataBindingUtil.setContentView(this, R.layout.camera_x);
        executor = ContextCompat.getMainExecutor(this);

        // Charger le modèle TensorFlow Lite  au démarrage
        try {
            model = MonModele.newInstance(this);
        } catch (IOException e) {
            e.printStackTrace();
            Toast.makeText(this, "Erreur lors du chargement du modèle", Toast.LENGTH_SHORT).show();
        }

        ListenableFuture<ProcessCameraProvider> cameraProviderFuture = ProcessCameraProvider.getInstance(this);
        cameraProviderFuture.addListener(() -> {
            try {
                cameraProvider = cameraProviderFuture.get();
                startCameraX();
            } catch (ExecutionException | InterruptedException e) {
                e.printStackTrace();
            }
        }, executor);
    }

    private void startCameraX() {
        CameraSelector cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA;
        Preview preview = new Preview.Builder().build();
        ImageAnalysis imageAnalysis = new ImageAnalysis.Builder()
                .setTargetResolution(new Size(200, 200)) // Résolution pour correspondre au modèle
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build();

        imageAnalysis.setAnalyzer(executor, imageProxy -> {
            // Obtenir le bitmap de l'image analysée
            Bitmap bitmap = imageProxyToBitmap(imageProxy);
            if (bitmap != null) {
                // Redimensionner l'image pour correspondre à l'entrée du modèle
                Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, 200, 200, true);
                fairePrediction(resizedBitmap);
            }
            imageProxy.close();
        });

        preview.setSurfaceProvider(binding.preview.getSurfaceProvider());

        try {
            cameraProvider.unbindAll();
            cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalysis);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @OptIn(markerClass = ExperimentalGetImage.class)
    private Bitmap imageProxyToBitmap(ImageProxy imageProxy) {
        Image image = imageProxy.getImage();
        if (image == null) {
            return null; // Si l'image est null, on retourne null directement
        }

        int width = image.getWidth();
        int height = image.getHeight();

        // Conversion en NV21
        YuvImage yuvImage = new YuvImage(extractNV21Data(image), ImageFormat.NV21, width, height, null);
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        yuvImage.compressToJpeg(new android.graphics.Rect(0, 0, width, height), 100, out);
        byte[] imageBytes = out.toByteArray();

        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
    }

    // Fonction pour extraire les données NV21 à partir de l'image
    private byte[] extractNV21Data(Image image) {
        ByteBuffer yBuffer = image.getPlanes()[0].getBuffer(); // Plan Y
        ByteBuffer uBuffer = image.getPlanes()[1].getBuffer(); // Plan U
        ByteBuffer vBuffer = image.getPlanes()[2].getBuffer(); // Plan V

        int ySize = yBuffer.remaining();
        int uSize = uBuffer.remaining();
        int vSize = vBuffer.remaining();

        byte[] nv21 = new byte[ySize + uSize + vSize];

        // Copie des données Y dans le tableau NV21
        yBuffer.get(nv21, 0, ySize);

        // Copie des données V et U dans le tableau NV21, après les données Y
        vBuffer.get(nv21, ySize, vSize);
        uBuffer.get(nv21, ySize + vSize, uSize);

        return nv21;
    }

    private void fairePrediction(Bitmap bitmap) {
        TensorImage tensorImage = new TensorImage(DataType.FLOAT32);
        tensorImage.load(bitmap);

        // Convertir l'image en ByteBuffer
        ByteBuffer byteBuffer = tensorImage.getBuffer();
        TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 200, 200, 3}, DataType.FLOAT32);
        inputFeature0.loadBuffer(byteBuffer);

        // Exécuter la prédiction
        MonModele.Outputs outputs = model.process(inputFeature0);
        TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();
        float[] scores = outputFeature0.getFloatArray();

        // Afficher le résultat
        String prediction;
        if (scores[0] < 0.5 && scores[1] < 0.5) {
            prediction = "Objet non identifié";
        } else {
            prediction = (scores[1] > scores[0]) ? "Rondoudou" : "Pikachu";
        }
        runOnUiThread(() -> binding.tv.setText("C'est un " + prediction));
        // Affichage temporaire pour vérifier les scores

    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (model != null) {
            model.close(); // Ferme le modèle pour libérer les ressources
            model = null;
        }
    }

}
