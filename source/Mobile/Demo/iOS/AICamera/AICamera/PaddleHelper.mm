//
//  PaddleHelper.m
//  AICamera
//
//  Created by Nicky Chan on 11/13/17.
//  Copyright © 2017 PaddlePaddle. All rights reserved.
//

#import <Foundation/Foundation.h>
#include "paddle_image_recognizer.h"
#include "PaddleHelper.h"

@implementation PaddleHelper : NSObject

+ (void)init_paddle {
    ImageRecognizer::init_paddle();
}

@end
