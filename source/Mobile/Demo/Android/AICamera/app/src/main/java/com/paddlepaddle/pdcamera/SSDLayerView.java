package com.paddlepaddle.pdcamera;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.util.AttributeSet;
import android.view.View;

import java.util.List;

/**
 * Created by Nickychan on 12/20/17.
 */

public class SSDLayerView extends View {

    private Paint mPaint = new Paint();
    private Paint mTextPaint = new Paint();
    private Paint mExteriorTextPaint = new Paint();
    private int mCornerRadiusInPx;
    private int mTextOffset;
    private int mScoreTextLength;
    private int mWidth;
    private int mHeight;
    private List<SSDData> mList;
    private boolean mDrawLabel;

    public SSDLayerView(final Context context, final AttributeSet attrs) {
        super(context, attrs);

        mTextPaint.setColor(Color.WHITE);
        mTextPaint.setStyle(Paint.Style.FILL);
        mTextPaint.setAntiAlias(false);
        mTextPaint.setTextSize(22 * getResources().getDisplayMetrics().density);
        mTextPaint.setFakeBoldText(true);

        mExteriorTextPaint.setColor(Color.BLACK);
        mExteriorTextPaint.setStyle(Paint.Style.STROKE);
        mExteriorTextPaint.setStrokeWidth(1 * getResources().getDisplayMetrics().density);
        mExteriorTextPaint.setAntiAlias(false);
        mExteriorTextPaint.setTextSize(22 * getResources().getDisplayMetrics().density);
        mExteriorTextPaint.setFakeBoldText(true);

        mPaint.setStyle(Paint.Style.STROKE);
        mPaint.setColor(Color.CYAN);
        int strokeWidthInDp = 3;
        int strokeWidthInPx = (int) (strokeWidthInDp * getResources().getDisplayMetrics().density);
        mPaint.setStrokeWidth(strokeWidthInPx);
        int cornerRadiusInDp = 10;
        mCornerRadiusInPx = (int) (cornerRadiusInDp * getResources().getDisplayMetrics().density);

        mTextOffset = (int) (3 * getResources().getDisplayMetrics().density);
        mScoreTextLength = (int) (40 * getResources().getDisplayMetrics().density);
    }

    public void setTextureViewDimen(int width, int height) {
        mWidth = width;
        mHeight = height;
    }

    public void populateSSDList(List<SSDData> list, boolean drawLabel) {
        mList = list;
        mDrawLabel = drawLabel;
        postInvalidate();
    }

    @Override
    public void draw(Canvas canvas) {
        super.draw(canvas);
        if (mList == null) return;

        for (SSDData ssdData : mList) {
            canvas.drawRoundRect(ssdData.xmin * mWidth, ssdData.ymin * mHeight,
                    ssdData.xmax * mWidth, ssdData.ymax * mHeight, mCornerRadiusInPx, mCornerRadiusInPx, mPaint);
            if (mDrawLabel) {
                //draw text
                canvas.drawText(ssdData.label, mTextOffset + ssdData.xmin * mWidth, ssdData.ymax * mHeight - mTextOffset, mTextPaint);
                canvas.drawText(ssdData.label, mTextOffset + ssdData.xmin * mWidth, ssdData.ymax * mHeight - mTextOffset, mExteriorTextPaint);
            }
            //draw score
            float roundedScore = Math.round(ssdData.accuracy * 100) / 100f;
            canvas.drawText(roundedScore + "", ssdData.xmax * mWidth - mScoreTextLength, ssdData.ymax * mHeight - mTextOffset, mTextPaint);
            canvas.drawText(roundedScore + "", ssdData.xmax * mWidth - mScoreTextLength, ssdData.ymax * mHeight - mTextOffset, mExteriorTextPaint);
        }

    }
}
