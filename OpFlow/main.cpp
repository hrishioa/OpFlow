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
#include <deque>

using namespace cv;
using namespace std;

//hide the local functions in an anon namespace
namespace {
    void help(char** av) {
        cout << "Add Man-page reference here.";
    }
    
    //VideoCapture function. Not currently used, but implemented for later real-time Optical Flow detection.
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
                    for(int i=0;i<30;i++)
                    {
                        capture >> frame;
                        sprintf(filename,"filename%.3d.jpg",i);
                        imwrite(filename,frame);
                        cout << "Saved " << filename << endl;
                        
                    }
                    sprintf(filename,"sfilename%.3d.jpg",n++);
                    imwrite(filename,frame);
                    cout << "Saved " << filename << endl;
                    break;
                default:
                    break;
            }
        }
        return 0;
    }
    
    deque<Point> findCorners(Mat img, int xarea, int yarea, int thres, bool verbose=true) {
        deque<Point> corners;
        
        //We're using Morevac Corner Detection here as it is easier to implement.
        //TODO: Consider using Harris Corners
        ofstream log; //This will be used for dumping raw data for corner analysis
        log.open("log.csv");
        log << "x,y,score1,score2\n";
        
        Mat outimg = img.clone(); //This will be used to provide a visual indication of the corners present
        
        if(verbose)
            printf("This is still under construction. - areas - (%d,%d), thres - %d",xarea,yarea,thres);
        
        int dimx = img.cols, dimy = img.rows;
        
        int count = 0;
        for(int startx=0;(startx+xarea)<dimx;startx+=xarea)
            for(int starty=0;(starty+yarea)<dimy;starty+=yarea)
            {
                count++;
                if(verbose)
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
                        if(verbose)
                            printf("Skipping due to dimensional or similarity issues");
                        continue;
                    }
                    Mat diff = abs(curarea-newarea);
                    results[dir%2] = mean(mean(diff))(0);
                }
                results[0]/=2;
                results[1]/=2;
                if(verbose)
                    printf("Scores obtained: %f, %f\n", results[0],results[1]);
                
                //thresholding
                if(results[0]>=thres && results[1]>=thres)
                {
                    corners.push_back(Point(startx,starty));
                    rectangle(outimg, Point(startx,starty), Point(startx+xarea,starty+yarea), Scalar(0),2);
                }
                
                log << startx << "," << starty << "," << results[0] << "," << results[1] << "\n";
            }
        log.close();
        
        if(verbose)
        {
            string winCorImg = "Corners found";
            namedWindow(winCorImg, WINDOW_AUTOSIZE);
            imshow(winCorImg, outimg);
            waitKey(0);
        }
        
        return corners;
    }
    
    Mat lucasKanade(Mat imgA, Mat imgB, int xsarea, int ysarea, int xarea, int yarea, deque<Point> corners, bool verbose=true, char filename[] = NULL)
    {
        Mat outimg = imgB.clone();
        
        int dimx = imgA.cols, dimy = imgA.rows;
        
        //iterate through each corner to find flow vectors
        for(int i=0;i<corners.size();i++)
        {
            Point cur_corner = corners[i];
            Point corner_mid = Point(corners[i].x+(int)(xarea/2),cur_corner.y+(int)(yarea/2));
            
            //Draw the corner in the out-image
            rectangle(outimg, cur_corner, cur_corner+Point(xarea,yarea), Scalar(0), 2);
            
            //Set range parameters
            Range sry = Range(max(0,corner_mid.y-(int)(ysarea/2)), min(dimy,corner_mid.y+(int)(ysarea/2)));
            Range srx = Range(max(0,corner_mid.x-(int)(xsarea/2)), min(dimx,corner_mid.x+(int)(xsarea/2)));
            
            //Now that we've found search windows, we can proceed to calculate Ix and Iy for each pixel in the search window
            //Ix
            Mat Ix = imgA.clone(), Iy=imgA.clone();
            
            int range=1;
            double G[2][2] = {0,0,0,0};
            double b[2] = {0,0};
            
            for(int x=srx.start; x<=srx.end; x++)
                for(int y=sry.start;y<=sry.end;y++)
                {
                    int px=x-range, nx=x+range;
                    int py=y-range, ny=y+range;
                    if(x==0) px=x;
                    if(x>=(dimx-1)) nx=x;
                    if(y==0) py=0;
                    if(y>=(dimy-1)) ny=y;
  
                    double curIx = ((int)imgA.at<uchar>(y,px)-(int)imgA.at<uchar>(y,nx))/2;
                    double curIy = ((int)imgA.at<uchar>(py,x)-(int)imgA.at<uchar>(ny,x))/2;
                    Ix.at<uchar>(y,x) = curIx;
                    Iy.at<uchar>(y,x) = curIy;
                    
                    //calculate G and b
                    G[0][0] += curIx*curIx;
                    G[0][1] += curIx*curIy;
                    G[1][0] += curIx*curIy;
                    G[1][1] += curIy*curIy;
                    
                    double curdI = ((int)imgA.at<uchar>(y,x)-(int)imgB.at<uchar>(y,x));
                    
                    b[0] += curdI*curIx;
                    b[1] += curdI*curIy;
                }
            
            //Since it's a royal pain-in-the-ass to download and link matrix libraries to my XCode
            //for just a single operation, we're just gonna do it by hand
            double detG = (G[0][0]*G[1][1])-(G[0][1]*G[1][0]);
            double Ginv[2][2] = {0,0,0,0};
            Ginv[0][0] = G[1][1]/detG;
            Ginv[0][1] = -G[0][1]/detG;
            Ginv[1][0] = -G[1][0]/detG;
            Ginv[1][1] = G[0][0]/detG;
            
            double V[2] = {Ginv[0][0]*b[0]+Ginv[0][1]*b[1],Ginv[1][0]*b[0]+Ginv[1][1]*b[1]};
            if(verbose)
                printf("\nFor the corner (%d,%d) - v = (%f,%f)",cur_corner.x,cur_corner.y,V[0],V[1]);
            
            line(outimg, corner_mid, corner_mid+Point(V[0]*10,V[1]*10), Scalar(1), 1);
        }
        string winCorImg = "Corners found";
        namedWindow(winCorImg, WINDOW_AUTOSIZE);
        imshow(winCorImg, outimg);
        waitKey(0);
        if(filename!=NULL)
        {
            cout << string("\nWriting lkoutput to o") << string(filename);
            imwrite(string("o")+string(filename), outimg);
        }
        return outimg;
    }
    
    Mat hornShunck(Mat imgA, Mat imgB, int scalefactor, int xysize, double smoothnessAssumption)
    {
        Mat outimg;
        
        //To improve performance and smoothness, we need to resize the image
        resize(imgA, imgA, Size(0,0), (double)scalefactor, (double)scalefactor);
        resize(imgB, imgB, Size(0,0), (double)scalefactor, (double)scalefactor);
        
        //Once again, we need to calculate the following variables - Ix, Iy and It
        
        return outimg;
    }
//end of namespace
}

int main(int ac, char** av) {
    
//    if (ac != 2) {
//        help(av);
//        return 1;
//    }
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
    
//    Step 1 - Implementing Corner Detection
//    Read the file
    Mat src = imread(av[1]);
    Mat src_color, img1, img2; //To store the file after conversion
    cvtColor(src, src_color, CV_BGR2GRAY);
    src = imread("filename000.jpg");
    cvtColor(src, img1, CV_BGR2GRAY);
    src = imread("filename001.jpg");
    cvtColor(src, img2, CV_BGR2GRAY);
    
    printf("Dimensions of the Image: %d, %d", src_color.cols, src_color.rows);
    
    for(;;)
    {
        cout << "\nPlease enter x,y parameters to read intensity (-1to begin corner detection):";
        int x,y;
        scanf("%d,%d",&x,&y);
        if (x<0 || y<0) break;
        int i = src_color.at<uchar>(y,x);
        printf("\n\nYou entered x - %d and y - %d.\nIntensity(%d,%d) = %d:", x,y,x,y,i);
    }
    
    int xarea=10,yarea=10,thres=8;
    
    cout << "\n Enter 1 to start Lucas-Kanade, 0 to skip: ";
    
    int inp=0;
    scanf("%d",&inp);
    if(inp==1)
    {
        //    Step 2 - Implementing Lucas-Kanade Tracker
        Mat previmg = img1, curimg;
        for(int i=1; i < 30;i++)
        {
            char buffer[17];
        
            sprintf(buffer, "filename%.3d.jpg", i-1);
            printf("\nLoaded %s for prev",buffer);
            src = imread(buffer);
            cvtColor(src, previmg, CV_BGR2GRAY);
        
            sprintf(buffer, "filename%.3d.jpg", i);
            printf("\nLoaded %s",buffer);
            src = imread(buffer);
            cvtColor(src, curimg, CV_BGR2GRAY);
        
            deque<Point> corn = findCorners(previmg, xarea, yarea, thres, false);
        
            lucasKanade(previmg, curimg, 3, 3, xarea, yarea, corn, true, buffer);
        }
    }

    
    //Next we try to implement Horn-Shunck
}