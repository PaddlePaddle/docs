//
//  ImageRecognizerPaddleWrapper.h
//  AICamera
//
//  Created by Nicky Chan on 11/9/17.
//  Copyright © 2017 PaddlePaddle. All rights reserved.
//

#ifndef ImageRecognizerPaddleWrapper_h
#define ImageRecognizerPaddleWrapper_h

#import <Foundation/Foundation.h>

@interface ImageRecognizerPaddleWrapper : NSObject

- (instancetype)initWithModel:(NSString *)modelFileName
               withNormHeight:(int)height
                withNormWidth:(int)width;
- (NSMutableArray *)inference:(unsigned char *)pixels
                   withHeight:(int)height
                    withWidth:(int)width
              withFilterScore:(float)filterScore;
- (void)destroy;

@end

#endif /* ImageRecognizerPaddleWrapper_h */
