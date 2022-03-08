# MrMOT
*Improving Multiple Object Tracking via Multiple Representations based Detection Fusion and Re-tracking*

The following figure shows our proposed MrMOT operation. The left side is the three baseline networks of MrMOT framework, and the right side is the output result of detection fusion and id re-tracking algorithms.

![Demo](https://github.com/Vio1etovo/MrMOT/blob/main/Demo.png?raw=true)


**Currently only one post-processing algorithm demo is released, and more related algorithms will be released in the future.**

## Installation
* Install dependencies:

    python-3.6

    numoy-1.19

    easydict-1.9

    [**optional**] opencv-python-4.5

## Data preparation
Download the examples from here. [Google Drive]

You need to download examples and replace the original folder

The example file package structure is as follows：
```
${MrMOT}
   └——————examples
           └——————data
                   └——————MOT17
                           └——————train
                                   └——————fairmot-MOT17-02
                                   └——————...
                           └——————test
                                   └——————fairmot-MOT17-01
                                   └——————...
                           └——————val
                                   └——————fairmot-MOT17-02
                                   └——————...
                   └——————MOT20
                           └——————...
           └——————res
                  └——————preprogress
                  └——————...
           
```
We put the tracking result text data of various methods in **data**, and put the results after algorithm fine-tuning in **res**

## Demo

Now we only provide the text data of the tracking results, you can visualize based on the text data yourself.

```
  python demo.py --cfg ./data.yaml --task train --sub_task MOT17  
```
You can change the task and sub_task to choose which data set to process, and we preset several methods to be processed. Of course, you can also choose to process your own text data, just set it in --methods.



## Acknowledgement
We used the preprocessed results of [FairMOT](https://github.com/ifzhang/FairMOT), [GSDT](https://github.com/yongxinw/GSDT), [TransTrack](https://github.com/PeizeSun/TransTrack), [TraDes](https://github.com/JialianW/TraDeS), [Deft](https://github.com/MedChaabane/DEFT) and other methods. Thanks for their wonderful works.


