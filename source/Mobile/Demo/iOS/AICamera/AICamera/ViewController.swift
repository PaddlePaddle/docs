//
//  ViewController.swift
//  SSDDemo
//
//  Created by Nicky Chan on 11/6/17.
//  Copyright © 2017 PaddlePaddle. All rights reserved.
//

import UIKit
import AVFoundation
import Foundation


class ViewController: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate {

    var captureSession : AVCaptureSession?
    var multiboxLayer : SSDMultiboxLayer?
    var previewLayer : AVCaptureVideoPreviewLayer?
    var captureDevice : AVCaptureDevice?

    var isRestarting = false;

    var imageRecognizer : ImageRecognizer?

    var timeStamp : TimeInterval?

    var index = 0

    //default settings
    var ssdModel : SSDModel = SSDModel.PascalMobileNet300
    var accuracyThreshold : Float = 0.5
    var minTimeInterval : Float = 0.3
    var backCamera = true

    @IBOutlet weak var settingsView: UIView!

    @IBOutlet weak var accuracyLabel: UILabel!
    @IBOutlet weak var timeRefreshLabel: UILabel!
    @IBOutlet weak var pascalMobileNetBtn: UIButton!
    @IBOutlet weak var pascalVgg300Btn: UIButton!
    @IBOutlet weak var faceMobileNetBtn: UIButton!
    @IBOutlet weak var backCameraBtn: UIButton!
    @IBOutlet weak var frontCameraBtn: UIButton!
    @IBOutlet weak var accuracySlider: UISlider!
    @IBOutlet weak var timeRefreshSlider: UISlider!


    @IBAction func pascalMobileNet300Click(_ sender: UIButton) {
        pendingRestartWithNewModel(newModel: SSDModel.PascalMobileNet300)
    }

    @IBAction func faceMobileNet300Click(_ sender: UIButton) {
        pendingRestartWithNewModel(newModel: SSDModel.FaceMobileNet160)
    }

    @IBAction func pascalVgg300Click(_ sender: UIButton) {
        pendingRestartWithNewModel(newModel: SSDModel.PascalVGG300)
    }

    @IBAction func backCameraClick(_ sender: UIButton) {
        pendingRestartWithCamera(backCamera: true)
    }

    @IBAction func frontCameraClick(_ sender: UIButton) {
        pendingRestartWithCamera(backCamera: false)
    }

    @IBAction func accurcyThresholdChanged(_ sender: UISlider) {

        accuracyThreshold = sender.value
        accuracyLabel.text = String.init(format: "%.02f", accuracyThreshold)
        let defaults = UserDefaults.standard
        defaults.set(accuracyThreshold, forKey: "accuracyThreshold")
    }

    @IBAction func timeRefreshChanged(_ sender: UISlider) {

        minTimeInterval = sender.value
        timeRefreshLabel.text = String.init(format: "%.02f", minTimeInterval)
        let defaults = UserDefaults.standard
        defaults.set(minTimeInterval, forKey: "timeRefresh")
    }

    func pendingRestartWithNewModel(newModel: SSDModel) {

        if ssdModel == newModel {
            return;
        }

        let defaults = UserDefaults.standard
        defaults.set(newModel.rawValue , forKey: "model")

        isRestarting = true
        ssdModel = newModel
    }


    func pendingRestartWithCamera(backCamera: Bool) {

        if self.backCamera == backCamera {
            return;
        }

        let defaults = UserDefaults.standard
        defaults.set(backCamera , forKey: "backCamera")

        isRestarting = true
        self.backCamera = backCamera
    }

    func restart() {
        //hack: just make it crash so that we can restart
        exit(0)
        DispatchQueue.main.async {
        self.timeStamp = nil
        self.index = 0;

        self.imageRecognizer?.release()
        self.imageRecognizer = ImageRecognizer(model: self.ssdModel)

        self.captureSession?.stopRunning()

        self.previewLayer?.removeFromSuperlayer()
        self.multiboxLayer?.removeFromSuperlayer()
        self.setupVideoCaptureAndStart()

        self.isRestarting = false
        }
    }

    func toggleSettings(_ sender:UITapGestureRecognizer){
        settingsView.isHidden = !settingsView.isHidden
    }

    override func viewDidLoad() {
        super.viewDidLoad()

        self.settingsView.isHidden = true

        checkModel()

        populateInitialSettings()

        let gesture = UITapGestureRecognizer(target: self, action:  #selector (self.toggleSettings (_:)))
        self.view.addGestureRecognizer(gesture)

        imageRecognizer = ImageRecognizer(model: ssdModel)

        setupVideoCaptureAndStart()
    }

    func checkModel() {
        var bundlePath = Bundle.main.bundlePath
        bundlePath.append("/")
        bundlePath.append(SSDModel.PascalVGG300.rawValue)
        pascalVgg300Btn.isHidden = !FileManager.default.fileExists(atPath: bundlePath)
    }

    func populateInitialSettings() {

        let defaults = UserDefaults.standard

        if let modelStr = defaults.string(forKey:"model") {
            self.ssdModel = SSDModel(rawValue: modelStr)!
        }
        var highlightBtn : UIButton?
        if ssdModel == SSDModel.FaceMobileNet160 {
            highlightBtn = faceMobileNetBtn;
        } else if ssdModel == SSDModel.PascalMobileNet300 {
            highlightBtn = pascalMobileNetBtn;
        } else if ssdModel == SSDModel.PascalVGG300 {
            highlightBtn = pascalVgg300Btn;
        }
        highlightBtn?.titleLabel?.font = UIFont.boldSystemFont(ofSize: 16)
        highlightBtn?.setTitleColor(self.view.tintColor, for: .normal)

        if let backCamera = defaults.object(forKey: "backCamera") {
            self.backCamera = backCamera as! Bool
        }

        if self.backCamera {
            backCameraBtn.titleLabel?.font = UIFont.boldSystemFont(ofSize: 16)
            backCameraBtn.setTitleColor(self.view.tintColor, for: .normal)
        } else {
            frontCameraBtn.titleLabel?.font = UIFont.boldSystemFont(ofSize: 16)
            frontCameraBtn.setTitleColor(self.view.tintColor, for: .normal)
        }

        if let accuracyThreshold = defaults.object(forKey: "accuracyThreshold") {
            self.accuracyThreshold = accuracyThreshold as! Float
            accuracySlider.setValue(self.accuracyThreshold, animated: false)
        }

        if let timeRefresh = defaults.object(forKey: "timeRefresh") {
            self.minTimeInterval = timeRefresh as! Float
            timeRefreshSlider.setValue(self.minTimeInterval, animated: false)
        }

        accuracyLabel.text = String.init(format: "%.02f", accuracyThreshold)
        timeRefreshLabel.text = String.init(format: "%.02f", minTimeInterval)

    }

    func setupVideoCaptureAndStart() {

        captureSession = AVCaptureSession()
        if let captureSession = captureSession {
        captureSession.sessionPreset = AVCaptureSessionPresetHigh

            captureDevice = AVCaptureDeviceDiscoverySession(deviceTypes: [AVCaptureDeviceType.builtInWideAngleCamera], mediaType: AVMediaTypeVideo, position: backCamera ? AVCaptureDevicePosition.back : AVCaptureDevicePosition.front).devices.first

        // setup video device input
        do {
            let videoDeviceInput: AVCaptureDeviceInput
            do {
                videoDeviceInput = try AVCaptureDeviceInput(device: captureDevice)
            }
            catch {
                fatalError("Could not create AVCaptureDeviceInput instance with error: \(error).")
            }

            captureSession.beginConfiguration()
            guard captureSession.canAddInput(videoDeviceInput) else {
                fatalError("CaptureSession can not add input")
            }
            captureSession.addInput(videoDeviceInput)
        }

        // setup preview
        let previewContainer = self.view.layer
        let previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)!
        previewLayer.frame = previewContainer.bounds
        previewLayer.contentsGravity = kCAGravityResizeAspect
        previewLayer.videoGravity = AVLayerVideoGravityResizeAspect
        previewContainer.insertSublayer(previewLayer, at: 0)
        self.previewLayer = previewLayer

        multiboxLayer = SSDMultiboxLayer()
        previewContainer.insertSublayer(multiboxLayer!, at: 1)

        // setup video output
        do {
            let videoDataOutput = AVCaptureVideoDataOutput()
            videoDataOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey: Int(kCVPixelFormatType_32BGRA)]
            videoDataOutput.alwaysDiscardsLateVideoFrames = true
            guard captureSession.canAddOutput(videoDataOutput) else {
                fatalError("CaptureSession can not add output")
            }
            captureSession.addOutput(videoDataOutput)

            captureSession.commitConfiguration()

            let queue = DispatchQueue(label: "com.paddlepaddle.SSDDemo")
            videoDataOutput.setSampleBufferDelegate(self, queue: queue)

            if let connection = videoDataOutput.connection(withMediaType: AVMediaTypeVideo) {
                if connection.isVideoOrientationSupported {
                    // Force recording to portrait
                    // use portrait does not work for some reason, try to rotate in c++ code instead
                    //                    connection.videoOrientation = .portrait
                }
            }
            captureSession.startRunning()
        }
        }
    }

    func captureOutput(_ output: AVCaptureOutput, didDrop sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
    }

    func captureOutput(_ output: AVCaptureOutput, didOutputSampleBuffer sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {

        if let ts = self.timeStamp {
            while(true) {
                if (NSDate().timeIntervalSince1970 >= Double(minTimeInterval) + ts) {
                    break;
                }
            }
        }

        index = index + 1
        if let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) {
            CVPixelBufferLockBaseAddress(imageBuffer, CVPixelBufferLockFlags(rawValue: 0))

            let width = CVPixelBufferGetWidth(imageBuffer)
            let height = CVPixelBufferGetHeight(imageBuffer)
            let baseAddress = CVPixelBufferGetBaseAddress(imageBuffer)

            let intBuffer = unsafeBitCast(baseAddress, to: UnsafeMutablePointer<UInt8>.self)

            CVPixelBufferUnlockBaseAddress(imageBuffer, CVPixelBufferLockFlags(rawValue: 0))

            let ssdDataList = imageRecognizer?.inference(imageBuffer: intBuffer, width: Int32(width), height: Int32(height), score: accuracyThreshold)

            self.timeStamp = NSDate().timeIntervalSince1970

            DispatchQueue.main.async {
                self.multiboxLayer?.displayBoxs(with: ssdDataList!, model:self.ssdModel, isBackCamera:self.backCamera)
            }
        }

        if (isRestarting) {
            restart()
        }
    }


    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()

        print("didReceiveMemoryWarning")
        // Dispose of any resources that can be recreated.
    }

}
