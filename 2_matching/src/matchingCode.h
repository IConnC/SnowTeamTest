#ifndef __MATCHINGCODE_H_INCLUDED__  
#define __MATCHINGCODE_H_INCLUDED__

#include <vector>
#include <string>
#include <filesystem>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

void matchPics(vector<filesystem::directory_entry> cam1Images, pair<int,int> cam1Size, 
               vector<filesystem::directory_entry> cam2Images, pair<int,int> cam2Size, 
               Mat fundamentalMatrix, string outputFile);

#endif