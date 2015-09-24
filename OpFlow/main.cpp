//
//  main.cpp
//  OpFlow
//
//  Created by Hrishi Olickel on 18/9/15.
//  Copyright (c) 2015 Hrishi Olickel. All rights reserved.
//

#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <fstream>
#include <math.h>
#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;

//hide the local functions in an anon namespace
namespace {
    void help(char** av) {
        cout << "The program captures frames from a video file, image sequence (01.jpg, 02.jpg ... 10.jpg) or camera connected to your computer." << endl
        << "Usage:\n" << av[0] << " <video file, image sequence or device number>" << endl
        << "q,Q,esc -- quit" << endl
        << "space   -- save frame" << endl << endl
        << "\tTo capture from a camera pass the device number. To find the device number, try ls /dev/video*" << endl
        << "\texample: " << av[0] << " 0" << endl
        << "\tYou may also pass a video file instead of a device number" << endl
        << "\texample: " << av[0] << " video.avi" << endl
        << "\tYou can also pass the path to an image sequence and OpenCV will treat the sequence just like a video." << endl
        << "\texample: " << av[0] << " right%%02d.jpg" << endl;
    }
    
    int process(VideoCapture& capture) {
        int n = 0;
        char filename[200];
        string window_name = "video | q or esc to quit. Ctrl+C to abort.";
        cout << "press space to save a picture. q or esc to quit" << endl;
        namedWindow(window_name, WINDOW_KEEPRATIO); //resizable window;
        Mat frame;
        
        for (;;) {
            capture >> frame;
            if (frame.empty())
                break;
            
            imshow(window_name, frame);
            char key = (char)waitKey(30); //delay N millis, usually long enough to display and capture input
            
            switch (key) {
                case 'q':
                case 'Q':
                case 27: //escape key
                    return 0;
                case ' ': //Save an image
                    sprintf(filename,"filename%.3d.jpg",n++);
                    imwrite(filename,frame);
                    cout << "Saved " << filename << endl;
                    break;
                default:
                    break;
            }
        }
        return 0;
    }
    
    void findCorners(Mat img, int xarea, int yarea) {
        //We're using Morevac Corner Detection here as it is easier to implement.
        //TODO: Consider using Harris Corners
        ofstream log; //This will be used for dumping raw data for corner analysis
        log.open("log.csv");
        
        log << "x,y,score1,score2\n";
        
        printf("This is still under construction.");
        
        int dimx = img.cols, dimy = img.rows;
        
        int count = 0;
        for(int startx=0;(startx+xarea)<dimx;startx+=xarea)
            for(int starty=0;(starty+yarea)<dimy;starty+=yarea)
            {
                count++;
                printf("\n Area %d - Currenty looking at area (%d-%d,%d-%d)\n",count,startx,startx+xarea,starty,starty+yarea);
                Mat curarea = img(Range(starty,min(starty+yarea,dimy)),Range(startx,min(dimx,startx+xarea)));
                double results[2] = {0,0};
                for(int dir = 0;dir<4;dir++)
                {
                    int newsx=startx,newsy=starty;
                    //Check similarity in each direction
                    switch(dir)
                    {
                        case 0: //left
                            newsx-=xarea;
                            newsx = max(newsx,0);
                            break;
                        case 1: //top
                            newsy-=yarea;
                            newsy = max(newsy,0);
                            break;
                        case 2: //right
                            newsx+=xarea;
                            newsx = min(newsx,dimx);
                            break;
                        case 3: //down
                            newsy+=yarea;
                            newsy = min(newsy,dimy);
                            break;
                        default:
                            break;
                    }
//                    printf("Current direction is %d. New area is (%d-%d,%d-%d)\n",dir,newsx,min(dimx,newsx+xarea),newsy,min(newsy+yarea,dimy));
                    
                    Mat newarea = img(Range(newsy,min(newsy+yarea,dimy)),Range(newsx,min(newsx+xarea,dimx)));

                    if(newarea.cols!=curarea.cols || newarea.rows!=curarea.rows)
                    {
                        printf("Skipping due to dimensional or similarity issues");
                        continue;
                    }
                    Mat diff = abs(curarea-newarea);
                    results[dir%2] = mean(mean(diff))(0);
                }
                results[0]/=2;
                results[1]/=2;
                printf("Scores obtained: %f, %f\n", results[0],results[1]);
                
                log << startx << "," << starty << "," << results[0] << "," << results[1] << "\n";
            }
        log.close();
    }
//end of namespace
}

int main(int ac, char** av) {
    
    if (ac != 2) {
        help(av);
        return 1;
    }
//    std::string arg = av[1];
//    VideoCapture capture(arg); //try to open string, this will attempt to open it as a video file or image sequence
//    if (!capture.isOpened()) //if this fails, try to open as a video camera, through the use of an integer param
//        capture.open(atoi(arg.c_str()));
//    if (!capture.isOpened()) {
//        cerr << "Failed to open the video device, video file or image sequence!\n" << endl;
//        help(av);
//        return 1;
//    }
//    return process(capture);
    
    //New Code being written
    //Read the file
    Mat src = imread(av[1]);
    Mat src_color; //To store the file after conversion
    cvtColor(src, src_color, CV_BGR2GRAY);
    
//    string windowname = "Main Window";
//    namedWindow(windowname, WINDOW_AUTOSIZE);
//    imshow(windowname, src_color);
//    waitKey(0);
    printf("Dimensions of the Image: %d, %d", src_color.cols, src_color.rows);
    
    for(;;)
    {
        cout << "\nPlease enter x,y parameters to read intensity:";
        int x,y;
        scanf("%d,%d",&x,&y);
        if (x<0 || y<0) break;
        int i = src_color.at<uchar>(y,x);
        printf("\n\nYou entered x - %d and y - %d.\nIntensity(%d,%d) = %d:", x,y,x,y,i);
    }
    findCorners(src_color, 20,20);
    
}