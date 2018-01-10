package com.example.dingjie.finaltextdetect;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.media.ExifInterface;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.support.annotation.NonNull;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AlertDialog;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.util.SparseArray;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;

import com.google.android.gms.vision.Frame;
import com.google.android.gms.vision.text.Text;
import com.google.android.gms.vision.text.TextBlock;
import com.google.android.gms.vision.text.TextRecognizer;
import com.googlecode.tesseract.android.TessBaseAPI;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity {
    ImageView imageView;
    TextView detectedTextView;
    Mat originalMat;
    static int REQUEST_READ_EXTERNAL_STORAGE = 0;
    static boolean read_external_storage_granted = false;
    private final int ACTION_PICK_PHOTO = 1;
    Bitmap textBitmap;
    private TessBaseAPI mTess; //Tess API reference
    String datapath = "";
    private BaseLoaderCallback mOpenCVCallBack = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                    //DO YOUR WORK/STUFF HERE
                    Log.i("OpenCV", "OpenCV loaded successfully.");
                    break;
                default:
                    super.onManagerConnected(status);
                    break;
            }
        }
    };
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        datapath = getFilesDir()+ "/tesseract/";
        checkFile(new File(datapath + "tessdata/"));
        String lang = "chi_tra";
        mTess = new TessBaseAPI();
        mTess.init(datapath, lang);
        imageView = (ImageView) findViewById(R.id.image);
        detectedTextView = (TextView) findViewById(R.id.detected_text);
        textBitmap = BitmapFactory.decodeResource(getResources(), R.drawable.cat);
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_3_0, this, mOpenCVCallBack);

        if (ContextCompat.checkSelfPermission(MainActivity.this, Manifest.permission.READ_EXTERNAL_STORAGE)
                != PackageManager.PERMISSION_GRANTED) {
            Log.i("permission", "request READ_EXTERNAL_STORAGE");
            ActivityCompat.requestPermissions(MainActivity.this, new String[]{Manifest.permission.READ_EXTERNAL_STORAGE},
                    REQUEST_READ_EXTERNAL_STORAGE);
        }else {
            Log.i("permission", "READ_EXTERNAL_STORAGE already granted");
            read_external_storage_granted = true;
        }
    }
    public void TestObjectDectect(View view){
        if(textBitmap != null) {
            Bitmap tempBitmap = textBitmap.copy(Bitmap.Config.ARGB_8888, true);
            originalMat = new Mat(tempBitmap.getHeight(),
                    tempBitmap.getWidth(), CvType.CV_8U);
            Utils.bitmapToMat(tempBitmap, originalMat);
        }
        Mat tmp_img = new Mat();
        boolean tmp = false;
        Mat grayMat = new Mat();
        Imgproc.cvtColor(originalMat, grayMat, Imgproc.COLOR_BGR2GRAY);
        Mat bin = new Mat();
        Imgproc.GaussianBlur(grayMat,grayMat,new Size(3,3),0);
        Imgproc.Canny(grayMat,bin,50,150);
        //Imgproc.GaussianBlur(bin,bin,new Size(3,3),0);
        //Imgproc.threshold(grayMat, bin, 128, 255,Imgproc.THRESH_BINARY);
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(bin,contours,hierarchy,Imgproc.RETR_LIST,Imgproc.CHAIN_APPROX_SIMPLE);
        double max = 0;
        ArrayList<Rect> arrayList = new ArrayList<>();
        ArrayList<Rect> arrayList2 = new ArrayList<>();

        int t = 0;
        if(!contours.isEmpty()) {
            for (int idx = 0; idx >= 0; idx = (int) hierarchy.get(0, idx)[0]) {
                MatOfPoint matOfPoint = contours.get(idx);
                Rect rect = Imgproc.boundingRect(matOfPoint);

                double width = rect.width;
                double height = rect.height;
                double ratio = width / height;

                boolean draw = true;
                if(arrayList.size()> 0 ) {
                    if (height > originalMat.rows() / 16) {
                        for(int i = 0 ; i < arrayList.size() ; i++) {
                            double aX = arrayList.get(i).x;
                            double aY = arrayList.get(i).y;
                            double aWidth = arrayList.get(i).width;
                            double aHeight = arrayList.get(i).height;

                            //all in
                            if (width + rect.x < aX + aWidth && rect.x > aX && height + rect.y < aY + aHeight && rect.y > aY) {
                                draw = false;
                            }
                            // half in
                            if (width + rect.x < aX + aWidth && rect.x > aX && height + rect.y < aY + aHeight && rect.y > aY) {
                                draw = false;
                            }
                            //bigger
                            if (rect.x <= aX && width + rect.x >= aX + aWidth && height + rect.y >= aY + aHeight && rect.y <= aY) {
                                arrayList.remove(i);
                                draw = true;
                            }


                        }
                    } else {
                        draw = false;
                    }
                }
                if(((ratio > 0.5 && ratio < 1.5) || (ratio >0.1 && ratio < 0.4))&& draw) {
                    arrayList.add(rect);
                }

            }
        }
        detectedTextView.setText("");
        ArrayList<Bitmap> bitmaps = new ArrayList<>();
        int row[] = new int[30];
        int r = 0;
        int currentSize = 0;
        for(int y = originalMat.rows()/3 ; y <= originalMat.rows() ;y+= originalMat.rows()/3) {
            row[r] = currentSize;
            r++;
            for (int x = 0; x < originalMat.cols(); x++) {
                for (int i = 0; i < arrayList.size(); i++) {
                    if (arrayList.get(i).x + arrayList.get(i).width == x && arrayList.get(i).y  <= y) {
                        Rect rect = arrayList.get(i);

                        Mat roi_img = new Mat(originalMat, rect);

                        Bitmap bitmap = Bitmap.createBitmap(roi_img.width(),roi_img.height(),textBitmap.getConfig());
                        Utils.matToBitmap(roi_img,bitmap);
                        bitmaps.add(bitmap);

                        arrayList2.add(rect);
                        arrayList.remove(i);
                        currentSize++;
                    }

                }
            }
        }
        int current2 = 0;
        int r2 = 0;
        for(int i = 0 ; i < bitmaps.size();i++){
            if(row[r2] == i){
                detectedTextView.append("\n");
                r2++;
            }
            String OCRresult = null;
            mTess.setImage(bitmaps.get(i));
            OCRresult = mTess.getUTF8Text();
            //TextView OCRTextView = (TextView) findViewById(R.id.OCRTextView);
            detectedTextView.append(OCRresult);
            Rect rect = arrayList2.get(i);
            Imgproc.rectangle(originalMat, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height), new Scalar(255, 0, 0), 2);

            Imgproc.putText(originalMat, String.valueOf(++t), new Point(rect.x, rect.y), 1, 1.5, new Scalar(120, 0, 20));


        }

        Bitmap processed = Bitmap.createBitmap(textBitmap.getWidth(), textBitmap.getHeight(), textBitmap.getConfig());
        Utils.matToBitmap(originalMat, processed);
        imageView.setImageBitmap(processed);
    }
    public void processImage(View view){

    }
    private void copyFiles() {
        try {
            //location we want the file to be at
            String filepath = datapath + "/tessdata/chi_tra.traineddata";

            //get access to AssetManager
            AssetManager assetManager = getAssets();

            //open byte streams for reading/writing
            InputStream instream = assetManager.open("tessdata/chi_tra.traineddata");
            OutputStream outstream = new FileOutputStream(filepath);

            //copy the file to the location specified by filepath
            byte[] buffer = new byte[1024];
            int read;
            while ((read = instream.read(buffer)) != -1) {
                outstream.write(buffer, 0, read);
            }
            outstream.flush();
            outstream.close();
            instream.close();

        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    private void checkFile(File dir) {
        //directory does not exist, but we can successfully create it
        if (!dir.exists()&& dir.mkdirs()){
            copyFiles();
        }
        //The directory exists, but there is no data file in it
        if(dir.exists()) {
            String datafilepath = datapath+ "/tessdata/chi_tra.traineddata";
            File datafile = new File(datafilepath);
            if (!datafile.exists()) {
                copyFiles();
            }
        }
    }
    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        getMenuInflater().inflate(R.menu.main_menu, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {

        int id = item.getItemId();

        //noinspection SimplifiableIfStatement
        if (id == R.id.open_gallery) {
            if (read_external_storage_granted) {
                Intent intent = new Intent(Intent.ACTION_PICK);
                intent.setType("image/*");
                startActivityForResult(intent, ACTION_PICK_PHOTO);
            } else {
                return true;
            }
        }
        return super.onOptionsItemSelected(item);
    }
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == ACTION_PICK_PHOTO && resultCode == RESULT_OK && null != data && read_external_storage_granted) {

            Uri selectedImage = data.getData();
            String[] filePathColumn = {MediaStore.Images.Media.DATA};
            Cursor cursor = getContentResolver().query(selectedImage, filePathColumn, null, null, null);
            String picturePath;
            if(cursor == null) {
                Log.i("data", "cannot load any image");
                return;
            }else {
                try {
                    cursor.moveToFirst();
                    int columnIndex = cursor.getColumnIndex(filePathColumn[0]);
                    picturePath = cursor.getString(columnIndex);
                }finally {
                    cursor.close();
                }
            }

            BitmapFactory.Options options = new BitmapFactory.Options();
            options.inSampleSize = 2;
            textBitmap = BitmapFactory.decodeFile(picturePath, options);

            int orientation = 0;
            try {
                ExifInterface imgParams = new ExifInterface(picturePath);
                orientation = imgParams.getAttributeInt(
                        ExifInterface.TAG_ORIENTATION,
                        ExifInterface.ORIENTATION_UNDEFINED);
            } catch (IOException e) {
                e.printStackTrace();
            }




                imageView.setImageBitmap(textBitmap);
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String permissions[], @NonNull int[] grantResults) {
        if (requestCode == REQUEST_READ_EXTERNAL_STORAGE) {
            // If request is cancelled, the result arrays are empty.
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                // permission was granted
                Log.i("permission", "READ_EXTERNAL_STORAGE granted");
                read_external_storage_granted = true;
            } else {
                // permission denied
                Log.i("permission", "READ_EXTERNAL_STORAGE denied");
            }
        }
    }
    public void detectText(View view) {
        detectedTextView.setText("");

        TextRecognizer textRecognizer = new TextRecognizer.Builder(this).build();
        if (!textRecognizer.isOperational()) {
            new AlertDialog.Builder(this)
                    .setMessage("Text recognizer could not be set up on your device :(")
                    .show();
            return;
        }

        Frame frame = new Frame.Builder().setBitmap(textBitmap).build();
        SparseArray<TextBlock> text = textRecognizer.detect(frame);

        for (int i = 0; i < text.size(); ++i) {
            TextBlock item = text.valueAt(i);
            if (item != null && item.getValue() != null) {
                List<? extends Text> textComponents = item.getComponents();
                for(Text currentText : textComponents) {
                    // Do your thing here
                    detectedTextView.append(currentText.getValue()+"\n");
                }
            }


        }
    }

}
