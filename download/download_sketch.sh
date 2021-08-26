cd ../../../
mkdir sketch_dataset
cd sketch_dataset
mkdir uncropped
cd uncropped
wget http://mmlab.ie.cuhk.edu.hk/archive/sketchdatabase/CUHK/training_88/Original_Images/CUHK_training_sketch.zip
wget http://mmlab.ie.cuhk.edu.hk/archive/sketchdatabase/CUHK/testing_100/Original_Images/CUHK_testing_sketch.zip
wget http://mmlab.ie.cuhk.edu.hk/archive/sketchdatabase/AR/Original_Images/AR_sketch.zip
wget http://mmlab.ie.cuhk.edu.hk/archive/sketchdatabase/XM2VTS/Original_Images/XM2VTS_sketch.zip

unzip CUHK_testing_sketch.zip
unzip CUHK_training_sketch.zip
unzip XM2VTS_sketch.zip
unzip AR_sketch.zip

cd ..
mkdir cropped
cd cropped
wget http://mmlab.ie.cuhk.edu.hk/archive/sketchdatabase/CUHK/training_88/Cropped_Images/CUHK_training_cropped_sketches.zip
wget http://mmlab.ie.cuhk.edu.hk/archive/sketchdatabase/CUHK/testing_100/Cropped_Images/CUHK_testing_cropped_sketches.zip
wget http://mmlab.ie.cuhk.edu.hk/archive/sketchdatabase/AR/Cropped_Images/AR_cropped_sketches.zip
wget http://mmlab.ie.cuhk.edu.hk/archive/sketchdatabase/XM2VTS/Cropped_Images/XM2VTS_cropped_sketches.zip

unzip CUHK_training_cropped_sketches.zip
unzip CUHK_testing_cropped_sketches.zip
unzip AR_cropped_sketches.zip
unzip XM2VTS_cropped_sketches.zip

cd ..
mkdir fiducial_points
cd fiducial_points
wget http://mmlab.ie.cuhk.edu.hk/archive/sketchdatabase/CUHK/training_88/Faducial_Points/CUHK_training_faducial_points_sketch.zip
wget http://mmlab.ie.cuhk.edu.hk/archive/sketchdatabase/CUHK/testing_100/Faducial_Points/CUHK_testing_faducial_points_sketch.zip
wget http://mmlab.ie.cuhk.edu.hk/archive/sketchdatabase/AR/Faducial_Points/faducial_points_sketch.zip
wget http://mmlab.ie.cuhk.edu.hk/archive/sketchdatabase/XM2VTS/Faducial_Points/Xm2VTS_faducial_points_sketch.zip

unzip CUHK_training_faducial_points_sketch.zip
unzip CUHK_testing_faducial_points_sketch.zip
unzip faducial_points_sketch.zip
unzip Xm2VTS_faducial_points_sketch.zip
