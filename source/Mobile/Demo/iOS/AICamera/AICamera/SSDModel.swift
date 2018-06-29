//
//  SSDModel.swift
//  AICamera
//
//  Created by Nicky Chan on 11/14/17.
//  Copyright © 2017 PaddlePaddle. All rights reserved.
//

import Foundation

enum SSDModel : String {
    case PascalMobileNet300 = "pascal_mobilenet_300_66.paddle"
    case FaceMobileNet160 = "face_mobilenet_160_91.paddle"
    case PascalVGG300 = "vgg_ssd_net.paddle"

    func normDimension() -> (Int32, Int32)
    {
        switch self
        {
            case .PascalMobileNet300:
                return (300, 300)
            case .FaceMobileNet160:
                return (160, 160)
            case .PascalVGG300:
                return (300, 300)
        }
    }
}
