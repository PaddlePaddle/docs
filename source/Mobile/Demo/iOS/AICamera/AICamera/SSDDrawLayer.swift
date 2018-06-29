//
//  SSDDrawLayer.swift
//  SSDDemo
//
//  Created by Nicky Chan on 11/7/17.
//  Copyright © 2017 PaddlePaddle. All rights reserved.
//

import UIKit

class SSDDrawLayer: CAShapeLayer {
    var labelLayer = CATextLayer()

    required override init() {
        super.init()
    }

    required init?(coder aDecoder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    func render(_ data: SSDData, model:SSDModel, isBackCamera:Bool) {

        let screenWidth = UIScreen.main.bounds.size.width
        let screenHeight = UIScreen.main.bounds.size.height

        let x = CGFloat(isBackCamera ? data.xmin : 1 - data.xmax) * screenWidth
        let y = CGFloat(data.ymin) * screenHeight
        let width = CGFloat(data.xmax - data.xmin) * screenWidth
        let height = CGFloat(data.ymax - data.ymin) * screenHeight

        if (model == SSDModel.FaceMobileNet160 && data.label != "aeroplane") {
            return;
        }

        //draw box
        self.path = UIBezierPath(roundedRect: CGRect(x: x, y: y, width: width, height: height), cornerRadius: 10).cgPath
        self.strokeColor = UIColor.cyan.cgColor
        self.lineWidth = 4.0
        self.fillColor = nil
        self.lineJoin = kCALineJoinBevel

        if (model == SSDModel.FaceMobileNet160) {
            //do not draw label for face
            return;
        }

        let text = String.init(format: "%@: %.02f", data.label, data.accuracy)
        var displayString = NSAttributedString(string: text, attributes: [
            NSStrokeColorAttributeName : UIColor.black,
            NSForegroundColorAttributeName : UIColor.white,
            NSStrokeWidthAttributeName : NSNumber(value: -6.0),
            NSFontAttributeName : UIFont.systemFont(ofSize: 20, weight: 3)
            ])

        //draw label

        labelLayer.string = displayString
        labelLayer.frame = CGRect.init(x: x + 4, y: y + height - 22, width: 1000, height: 30)
        addSublayer(labelLayer)
    }
}
