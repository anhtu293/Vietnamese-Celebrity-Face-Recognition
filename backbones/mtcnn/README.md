# MTCNN face detection and alignment

## Introduction

  This is a mxnet implementation of [Zhang](https://kpzhang93.github.io/)'s work: Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Neural Networks. It's fast and accurate,  see [link](https://github.com/kpzhang93/MTCNN_face_detection_alignment). This implementation of MTCNN should have **almost** the same output with the original work.  


## Environment

This repository has been tested under the following environment:

-   Python 2.7 
-   Ubuntu 18.04
-   Mxnet-cu90 (==1.3.0)

## Testing

-   Use `python main.py` to test this detection and alignment method.

-   You can change `ctx` to `mx.gpu(0)` to use GPU for faster detection.


see `mtcnn_detector.py` for the details about the parameters. this function use [dlib](http://dlib.net/)'s align strategy, which works well on profile images :) 

## Results

![Detetion Results](https://raw.githubusercontent.com/deepinx/mtcnn-face-detection/master/sample-images/detection_result.png)



## License

MIT LICENSE



## Reference
```
@article{Zhang2016Joint,
  title={Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks},
  author={Zhang, Kaipeng and Zhang, Zhanpeng and Li, Zhifeng and Yu, Qiao},
  journal={IEEE Signal Processing Letters},
  volume={23},
  number={10},
  pages={1499-1503},
  year={2016},
}
```

## Acknowledgment

The code is adapted based on an intial fork from the [mxnet_mtcnn_face_detection](https://github.com/YYuanAnyVision/mxnet_mtcnn_face_detection) repository.
