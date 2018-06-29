//
//  SSDData.h
//  AICamera
//
//  Created by Nicky Chan on 11/11/17.
//  Copyright © 2017 PaddlePaddle. All rights reserved.
//

#import <Foundation/Foundation.h>

@interface SSDData : NSObject

@property(nonatomic) NSString *label;
@property(nonatomic) float accuracy;
@property(nonatomic) float xmin;
@property(nonatomic) float ymin;
@property(nonatomic) float xmax;
@property(nonatomic) float ymax;

@end
