//#pragma GCC diagnostic ignored "-Wdeprecated-enum-enum-conversion" // opencv uses enum conversion a lot. the warnings are distracting

#include <iostream>
#include <experimental/filesystem>
#include <cstdint>
#include <vector>
#include <string>
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include "util/timer.h"
#include <sstream>
#include <fstream>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/photo.hpp>
#include "util/savepng.h"
#include <memory>
#include "crop_filter/crop_filter.h"

// shorten filesystem namespace
namespace fs = std::experimental::filesystem;

static const std::string CLI_HELP_STR = R"(Usage: ./snowflake_classifier <mode> [arguments...]
Modes:
	daemon <input_folder> bind_address
		Run continuously and process new images (TODO).

	batch <input_folder> <output_folder> [masc|smas]
		Process images in input_folder and write results to output_folder.
)";

static void saveImage(const cv::Mat& img, 
					  const std::string& folder, 
					  std::string fileStem, 
					  int num, 
					  const std::pair<int, int>& c, 
					  float sharpness);

// Uploads image to GPU, crops the image, and saves to filesystem
static void processImage(const fs::path& filepath, std::unique_ptr<crop_filter>& crop_filter_instance, cv::Mat& image, timer& t,
						 const std::string& usable_folder, const std::string& blurry_folder, const std::string& oversize_folder) {

	image = cv::imread(filepath.string(), cv::IMREAD_GRAYSCALE);
	
	// crude SMAS noise cutoff
	// cv::threshold(image, image, 120, 255, cv::THRESH_TOZERO); // 2-2023 images
	// cv::threshold(image, image, 50, 255, cv::THRESH_TOZERO); // 1-2022 images
	
	// some of the SMAS cameras needs these to block out part of the background
	
	// CAM0_1
	// cv::rectangle(foreground, {0, 700}, {450, 1450}, {0, 255, 255}, cv::FILLED);
	// cv::rectangle(foreground, {300, 500}, {400, 750}, {0, 0, 0}, cv::FILLED);
	
	// CAM1_1
	// cv::rectangle(image, {0, 1500}, {0, 350}, {0, 255, 255}, cv::FILLED);
	
	cv::cuda::GpuMat image_gpu;
	image_gpu.upload(image);
	
	std::cout << "Loading " << filepath << '\n';
	
	// the image can be empty sometimes??
	if (image.empty()) {
		return;
	}
	
	// create crop_filter instance if it doesn't exist yet or if it doesn't match image_gpu size
	// if ((crop_filter_instance == nullptr) || (crop_filter_instance->get_width() != (u32) image_gpu.cols) || (crop_filter_instance->get_height() != (u32) image_gpu.rows)) {
	if (crop_filter_instance == nullptr) {
		crop_filter_instance = std::make_unique<crop_filter>(image_gpu.cols, image_gpu.rows);
		// disable spectreal sharpness calculation
		crop_filter_instance->enable_spectral(false);
		// without spectral the threshold needs to be a bit lower
		crop_filter_instance->set_blur_threshold(0.13f);
	}
	
	// reset timer
	t.discardTime();
	
	unsigned int number_sharp_flakes = crop_filter_instance->crop(image_gpu);
	
	// print time for this iteration
	t.lap();
	
	// skip saving anything if there are no sharp flakes
	// if (number_sharp_flakes == 0) {
	// 	continue;
	// }
	
	uint32_t snowflake_number = 0;
	
	// save sharp flake images
	const std::vector<cv::Mat>& sharp_flakes = crop_filter_instance->get_cropped_images();
	const std::vector<std::pair<u32, u32>>& sharp_coords = crop_filter_instance->get_cropped_coords();
	const std::vector<f32>& sharp_sharpness = crop_filter_instance->get_cropped_sharpness();
	for (uint32_t i = 0; i < sharp_flakes.size(); ++i) {
		saveImage(sharp_flakes[i], usable_folder, filepath.stem(), snowflake_number + i, sharp_coords[i], sharp_sharpness[i]);
	}
	snowflake_number += sharp_flakes.size();
	
	// save blurry flake images
	const std::vector<cv::Mat>& blurry_flakes = crop_filter_instance->get_blurry_images();
	const std::vector<std::pair<u32, u32>>& blurry_coords = crop_filter_instance->get_blurry_coords();
	const std::vector<f32>& blurry_sharpness = crop_filter_instance->get_blurry_sharpness();
	for (uint32_t i = 0; i < blurry_flakes.size(); ++i) {
		saveImage(blurry_flakes[i], blurry_folder, filepath.stem(), snowflake_number + i, blurry_coords[i], blurry_sharpness[i]);
	}
	snowflake_number += blurry_flakes.size();
	
	// save oversize flake images
	const std::vector<cv::Mat>& oversize_flakes = crop_filter_instance->get_oversize_images();
	const std::vector<std::pair<u32, u32>>& oversize_coords = crop_filter_instance->get_oversize_coords();
	for (uint32_t i = 0; i < oversize_flakes.size(); ++i) {
		saveImage(oversize_flakes[i], oversize_folder, filepath.stem(), snowflake_number + i, oversize_coords[i], 0);
	}

	// image_gpu.release();
}

// Processes images saved on the filesystem rather than in memory
// If isSMAS is true then run SMAS processing, else run MASC
static void runBatchPipeline(const std::string& input_path, const std::string& output_folder, bool isSMAS) {
	fs::create_directory(output_folder);

	// Process for SMAS or MASC Cameras
	int num_cameras = isSMAS ? 7 : 3;
	char camera_name_list[7] = {'0', '1', '2', '3', '4', '5', '6'};

	char camera_name;
	for (int camIndex = 0; camIndex < num_cameras; camIndex++) {
		camera_name = camera_name_list[camIndex];

		std::string usable_folder   = output_folder + "/usable_images";
		std::string unusable_folder = output_folder + "/unusable_images";
		std::string oversize_folder = output_folder + "/unusable_images/oversize_images";
		std::string blurry_folder   = output_folder + "/unusable_images/blurry_images";
		
		fs::create_directory(unusable_folder);
		fs::create_directory(oversize_folder);
		fs::create_directory(blurry_folder);
		fs::create_directory(usable_folder);
		
		oversize_folder.append("/").push_back(camera_name);
		blurry_folder.append("/").push_back(camera_name);
		usable_folder.append("/").push_back(camera_name);

		fs::create_directory(oversize_folder);
		fs::create_directory(blurry_folder);
		fs::create_directory(usable_folder);
		
		std::vector<fs::path> paths;
		cv::Mat image;
		std::unique_ptr<crop_filter> crop_filter_instance = nullptr;
		
		fs::path filepath;
		std::string filename;

		for (const fs::directory_entry& entry : fs::recursive_directory_iterator{input_path}) {
			filepath = entry.path();
			filename = filepath.filename().string();

			if (fs::is_directory(filepath)) {
				continue;
			}
			
			if (filename.rfind("2024", 0) != 0) continue; // Temp for 2024 processing

			size_t camera_pos = filename.find(".");
			if (camera_pos == std::string::npos || camera_pos < 2) {
				continue;
			}
			
			if (filename.at(camera_pos - 2) != camera_name) {
				continue;
			}

			paths.emplace_back(filepath);
		}
		
		// sort input images so background subtraction works
		std::sort(paths.begin(), paths.end());
		
		// timer to test how long things take
		timer t;
		
		for (const fs::path& filepath : paths) {
			processImage(filepath, crop_filter_instance, image, t, usable_folder, blurry_folder, oversize_folder);
		}

		crop_filter_instance.reset(); // Reset after every camera, some cameras have different resolutions
	}
}

static void testPipeline(const std::string& input_path, const std::string& output_folder) {

}


int main(int argc, char** argv) {
	
	// opencv likes to print a lot of warnings sometimes
	cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_SILENT);
	
	if (argc < 4) {
		std::cout << "Unable to parse arguments!\n";
		std::cout << CLI_HELP_STR;
		std::exit(-1);
	}
	
	std::string mode = argv[1];
	std::string input_folder = argv[2];
	std::string output_folder = argv[3];

	if (cv::cuda::getCudaEnabledDeviceCount() < 1) {
		std::cout << "No CUDA capable device found! aborting!\n";
		std::exit(-1);
	}

	// -- TEMPORARY FOR PROCESSING --
	if (cv::cuda::getCudaEnabledDeviceCount() > 1) {
		cv::cuda::setDevice(1);
	}
	// -- TEMPORARY END --

	fs::path input_path{input_folder};
	
	if (!fs::is_directory(input_path)) {
		std::cout << "input folder " << input_folder << " not found! aborting!\n";
		std::exit(-1);
	}
	
	if (mode == "daemon") {
		std::cout << "Not Implemented!\n";
		return 1;
	} else if (mode == "batch") {
		if (argc < 5) {
			std::cout << CLI_HELP_STR;
			return 1;
		}
		std::string smas_string = argv[4];
		bool isSMAS;
		if (smas_string == "smas") isSMAS = true;
		else if (smas_string == "masc") isSMAS = false;
		else {
			std::cout << CLI_HELP_STR;
			return 1;
		}

		runBatchPipeline(input_path, output_folder, isSMAS);
	} else if (mode == "test") {
		testPipeline(input_path, output_folder);
	} else {
		std::cout << CLI_HELP_STR;
		return 1;
	}
	
	return 0;
}

static void saveImage(const cv::Mat& img, const std::string& folder, std::string fileStem, int num, const std::pair<int, int>& c, float sharpness) {
	int cY = c.first;
	int cX = c.second;

	// Place cropped number between brackets in filename (01 02 03 ... 09 10 11 12 13 ...)
	std::string croppedNum;
	if (num < 10) croppedNum.append("0");
	croppedNum.append(std::to_string(num));

	int beginBracket = fileStem.find("[");
	int endBracket = fileStem.find("]");

	fileStem = fileStem.substr(0, beginBracket + 1) + croppedNum + fileStem.substr(endBracket);

	// Concat 
	std::string outPath = folder + "/" + fileStem + 'X' + std::to_string(cX) + 'Y' + std::to_string(cY) + "_S" + std::to_string(sharpness) + ".png";

	// std::cout << "Saving to " << outPath << '\n';
	static std::vector<int> compression_params{cv::IMWRITE_PNG_COMPRESSION, 9};
	cv::imwrite(outPath, img, compression_params);
}

