package com.paddlepaddle.pdcamera;

/**
 * Created by Nickychan on 1/9/18.
 */

public enum SSDModel {

    PASCAL_MOBILENET_300("pascal_mobilenet_300_66.paddle", 300, 300),
    FACE_MOBILENET_160("face_mobilenet_160_91.paddle", 160, 160);

    public static float[] MEANS = {104, 117, 124};

    public static final String[] LABELS = {
            "background" , "aeroplane", "bicycle"  , "background" ,
            "boat"       , "bottle"   , "bus"      , "car"        ,
            "cat"        , "chair"    , "cow"      , "diningtable",
            "dog"        , "horse"    , "motorbike", "person"     ,
            "pottedplant", "sheep"    , "sofa"     , "train"      ,
            "tvmonitor" };

    public String modelFileName;
    public int width;
    public int height;

    SSDModel(String modelFileName, int width, int height) {
        this.modelFileName = modelFileName;
        this.width = width;
        this.height = height;
    }

    public static SSDModel fromModelFileName(String modelFileName) {

        for (SSDModel model : SSDModel.values()) {
            if (model.modelFileName.equals(modelFileName)) {
                return model;
            }
        }
        return null;
    }

}
