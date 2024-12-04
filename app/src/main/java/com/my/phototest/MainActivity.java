package com.my.phototest;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.ImageDecoder;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.databinding.DataBindingUtil;

import com.my.phototest.databinding.ActivityMainBinding;
import com.my.phototest.ml.MonModele;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;

public class MainActivity extends AppCompatActivity {

    ActivityMainBinding binding;
    private Bitmap img;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = DataBindingUtil.setContentView(this, R.layout.activity_main);

        binding.cadrePhoto.setImageResource(R.drawable.all_pokemon);
        binding.prendrePhoto.setOnClickListener(view -> {
            Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
            intent.setType("image/*");
            startActivityForResult(intent, 100);
        });

        binding.cam.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                startActivity(new Intent(getApplicationContext(), CameraX.class));
            }
        });

        binding.fairePrediction.setOnClickListener(view -> {
            // Vérifie si l'image est chargée avant de tenter la prédiction
            if (img != null) {
                // Redimensionner l'image pour la rendre de 200x200 pixels
                img = Bitmap.createScaledBitmap(img, 200, 200, true);

                try {
                    MonModele model = MonModele.newInstance(getApplicationContext());

                    // Créer un objet TensorImage à partir de l'image redimensionnée
                    TensorImage tensorImage = new TensorImage(DataType.FLOAT32);
                    tensorImage.load(img);

                    // Convertir l'image en un ByteBuffer pour l'entrée du modèle
                    ByteBuffer byteBuffer = tensorImage.getBuffer();

                    // Créer l'entrée pour le modèle (taille attendue : 1, 200, 200, 3)
                    TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 200, 200, 3}, DataType.FLOAT32);
                    inputFeature0.loadBuffer(byteBuffer);

                    // Exécuter la prédiction
                    MonModele.Outputs outputs = model.process(inputFeature0);
                    TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

                    // Récupérer les résultats sous forme de tableau de scores
                    float[] scores = outputFeature0.getFloatArray();

                    // Trouver l'indice de la classe avec la probabilité la plus élevée
                    int predictedClass = scores[0] > scores[1] ? 1 : 0;

                    // Afficher la classe prédite dans le TextView
                    String prediction = (predictedClass == 1) ? "Pikachu" : "Rondudu";
                    binding.tv.setText("C'est un " + prediction);

                    model.close();
                } catch (IOException e) {
                    e.printStackTrace();
                    binding.tv.setText("Erreur de prédiction");
                }
            } else {
                binding.tv.setText("Aucune image sélectionnée");
            }
        });
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (requestCode == 100 && resultCode == RESULT_OK && data != null) {
            // Mettre à jour l'image de la photo dans le cadre
            Uri uri = data.getData();
            binding.cadrePhoto.setImageURI(uri);

            try {
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) { // API 28 et plus
                    img = ImageDecoder.decodeBitmap(ImageDecoder.createSource(this.getContentResolver(), uri));
                } else { // Pour les versions inférieures à API 28
                    img = MediaStore.Images.Media.getBitmap(this.getContentResolver(), uri);
                }

                // Convertir l'image en ARGB_8888
                if (img.getConfig() != Bitmap.Config.ARGB_8888) {
                    img = img.copy(Bitmap.Config.ARGB_8888, true);
                }

            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
}
