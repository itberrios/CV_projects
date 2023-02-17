#include <iostream>
#include <math.h>
#include <algorithm>
#include <random>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


// remapping an image by creating wave effects
void wave(const cv::Mat &image, cv::Mat &result) {

	// the map functions
	cv::Mat srcX(image.rows, image.cols, CV_32F); // x-map
	cv::Mat srcY(image.rows, image.cols, CV_32F); // y-map

	// creating the mapping
	for (int i=0; i <image.rows; i++) {
		for (int j=0; j<image.cols; j++) {
            // wave
			srcX.at<float>(i,j) = j;
			srcY.at<float>(i,j) = i + 3*sin(j/6.0);
		}
	}

	// applying the mapping
	cv::remap(image,  // source image
		      result, // destination image
			  srcX,   // x map
			  srcY,   // y map
			  cv::INTER_LINEAR); // interpolation method
}


// remapping an image by creating jitter effects
// use a template to pass in a Random Number generator
// in C++ 20 , we could probably just use the auto keyword
template <class RNG> 
void jitter(const cv::Mat &image, cv::Mat &result, RNG &noise) {
	// the map functions
	cv::Mat srcX(image.rows, image.cols, CV_32F); // x-map
	cv::Mat srcY(image.rows, image.cols, CV_32F); // y-map

	// creating the mapping
	for (int i=0; i <image.rows; i++) {  // rows --> y-axis
		for (int j=0; j<image.cols; j++) { // cols --> x-axis
            
            // jitter
            srcX.at<float>(i,j) = j + noise();
			srcY.at<float>(i,j) = i + noise(); 

		}
	}

	// applying the mapping
	cv::remap(image,  // source image
		      result, // destination image
			  srcX,   // x map
			  srcY,   // y map
			  cv::INTER_LINEAR); // interpolation method
}


// // scarmble image
// void scramble(const cv::Mat &image, cv::Mat &result) {

//     // get number of rows and columns
//     int nrows = image.rows;
//     int ncols = image.cols;

//     // the map functions
// 	cv::Mat srcX(image.rows, image.cols, CV_32F); // x-map
// 	cv::Mat srcY(image.rows, image.cols, CV_32F); // y-map

// 	// 1) create vectors from 0 to nrows and another from 0 to ncols
//     std::vector<int> rowVecShuffled(nrows);
//     std::vector<int> colVecShuffled(ncols);
    
//     std::iota (std::begin(rowVecShuffled), std::end(rowVecShuffled), 0); 
//     std::iota (std::begin(colVecShuffled), std::end(colVecShuffled), 0);
    
//     // 2) shuffle these vectors
//     auto rng = std::default_random_engine {};
//     std::shuffle(std::begin(rowVecShuffled), std::end(rowVecShuffled), rng);
//     std::shuffle(std::begin(colVecShuffled), std::end(colVecShuffled), rng);

//     std::cout << "nrows: " << nrows << " ncols: " << ncols << std::endl;
//     std::cout << "rowVec size: " << rowVecShuffled.size() 
//               << " colVec size: " << colVecShuffled.size() << std::endl;

//     // 3) map regular locations to shuffled locations
//     // creating the mapping
// 	for (int i=0; i<image.rows; i++) {  // rows --> y-axis
// 		for (int j=0; j<image.cols; j++) { // cols --> x-axis
            
//             // std::cout << "( " << i << ", " << j << " )" << std::endl;
//             srcX.at<float>(i,j) = colVecShuffled.at(j);
// 			srcY.at<float>(i,j) = rowVecShuffled.at(i); 

// 		}
// 	}

// 	// applying the mapping
// 	cv::remap(image,  // source image
// 		      result, // destination image
// 			  srcX,   // x map
// 			  srcY,   // y map
// 			  cv::INTER_LINEAR); // interpolation method
// }


void scramble(const cv::Mat &image, cv::Mat &result, 
        std::vector<int> rowVec, std::vector<int> colVec) {

    // the map functions
	cv::Mat srcX(image.rows, image.cols, CV_32F); // x-map
	cv::Mat srcY(image.rows, image.cols, CV_32F); // y-map

	// 1) fill vectors from 0 to nrows and another from 0 to ncols
    std::iota (std::begin(rowVec), std::end(rowVec), 0); 
    std::iota (std::begin(colVec), std::end(colVec), 0);
    
    // 2) shuffle these vectors
    auto rng = std::default_random_engine {};
    std::shuffle(std::begin(rowVec), std::end(rowVec), rng);
    std::shuffle(std::begin(colVec), std::end(colVec), rng);

    // std::cout << "nrows: " << nrows << " ncols: " << ncols << std::endl;
    // std::cout << "rowVec size: " << rowVec.size() 
    //           << " colVec size: " << colVec.size() << std::endl;

    // 3) map regular locations to shuffled locations
    // creating the mapping
	for (int i=0; i<image.rows; i++) {  // rows --> y-axis
		for (int j=0; j<image.cols; j++) { // cols --> x-axis
            
            // std::cout << "( " << i << ", " << j << " )" << std::endl;
            srcX.at<float>(i,j) = colVec.at(j);
			srcY.at<float>(i,j) = rowVec.at(i); 
		}
	}

	// applying the mapping
	cv::remap(image,  // source image
		      result, // destination image
			  srcX,   // x map
			  srcY,   // y map
			  cv::INTER_LINEAR); // interpolation method
}


void unscramble(const cv::Mat &image, cv::Mat &result, 
        std::vector<int> rowVecShuffled, std::vector<int> colVecShuffled) {

    // the map functions
	cv::Mat srcX(image.rows, image.cols, CV_32F); // x-map
	cv::Mat srcY(image.rows, image.cols, CV_32F); // y-map

	// get row and col vectors
	std::vector<int> rowVec(image.rows);
    std::vector<int> colVec(image.cols);
	std::iota (std::begin(rowVec), std::end(rowVec), 0); 
    std::iota (std::begin(colVec), std::end(colVec), 0);

    // creating the mapping
	// for (int i=0; i<image.rows; i++) {  // rows --> y-axis
	// 	for (int j=0; j<image.cols; j++) { // cols --> x-axis
            
    //         // std::cout << "( " << i << ", " << j << " )" << std::endl;
    //         // srcX.at<float>(rowVec.at(i), colVec.at(j)) = j;
	// 		// srcY.at<float>(rowVec.at(i), colVec.at(j)) = i; 

    //         // srcX.at<float>(i, colVec.at(j)) = j;
	// 		// srcY.at<float>(rowVec.at(i), j) = i; 

    //         // srcX.at<float>(i, j) = colVec.at(j);
	// 		// srcY.at<float>(i, j) = rowVec.at(i); 

    //         // srcX.at<float>(rowVec.at(i), colVec.at(j)) = colVec.at(j);
	// 		// srcY.at<float>(rowVec.at(i), colVec.at(j)) = rowVec.at(i); 

    //         srcX.at<float>(i, j) = colVec.at(colVecShuffled.at(j));
	// 		srcY.at<float>(i, j) = i;

	// 	}
	// }

	for (int i: rowVecShuffled) {
		for (int j: colVecShuffled) {
			srcX.at<float>(i, j) = colVec.at(j);
			srcY.at<float>(i, j) = rowVec.at(i);
		}
	}

	// applying the mapping
	cv::remap(image,  // source image
		      result, // destination image
			  srcX,   // x map
			  srcY,   // y map
			  cv::INTER_LINEAR); // interpolation method
}



int main(int, char**) {
    cv::Mat image = cv::imread("../boldt.jpg");

    cv::namedWindow("Image");
	cv::imshow("Image", image);

    // Define random generator 
    std::default_random_engine generator;

    // Additive White Gaussian Noise
    const double mean = 0.0;
    const double stddev = 1.0;
    // type std::_Bind<std::normal_distribution<float> (std::mt19937)>
    auto awg_noise = std::bind(std::normal_distribution<float>{mean, stddev},
                               std::mt19937(std::random_device{}()));

    // Uniform Noise
    auto noise = std::bind(std::uniform_real_distribution<float>{-5, 5},
                               std::mt19937(std::random_device{}()));

	// remap image
	cv::Mat result;
	// wave(image, result);
    // jitter(image, result, awg_noise);

    // scramble image
    std::vector<int> rowVecShuffled(image.rows);
    std::vector<int> colVecShuffled(image.cols);
    scramble(image, result, rowVecShuffled, colVecShuffled);

    // unscramble image
    cv::Mat unscrambled;
    unscramble(image, unscrambled, rowVecShuffled, colVecShuffled);

    // now explore the difference between the orginal and remapped image
    cv::Mat combined;
    cv::addWeighted(image, 1.5, result, -0.5, 0.0, combined);

	cv::namedWindow("Remapped image");
	cv::imshow("Remapped image", result);

    cv::namedWindow("Combined");
    cv::imshow("Combined", combined);

    cv::namedWindow("Unscrambled");
    cv::imshow("Unscrambled", unscrambled);

	cv::waitKey();
    cv::destroyAllWindows();
}