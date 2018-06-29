//
//  ImageRecognizer.swift
//  SSDDemo
//
//  Created by Nicky Chan on 11/7/17.
//  Copyright © 2017 PaddlePaddle. All rights reserved.
//

import Foundation

protocol ImageRecognizerDelegate {
    func imageRecognizedSuccess(_ result: SSDData)
    func imageRecognizedError()
}

class ImageRecognizer {

    var imageRecognizer: ImageRecognizerPaddleWrapper?

    init(model: SSDModel) {
        imageRecognizer = ImageRecognizerPaddleWrapper(model: model.rawValue, withNormHeight: model.normDimension().0, withNormWidth: model.normDimension().1)
    }

    func inference(imageBuffer: UnsafeMutablePointer<UInt8>!, width: Int32, height: Int32, score: Float) -> NSMutableArray! {

        return imageRecognizer?.inference(imageBuffer, withHeight: height, withWidth: width, withFilterScore: score)
    }

    func release() {
        imageRecognizer?.destroy()
    }

}
