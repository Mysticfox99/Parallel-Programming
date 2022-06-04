#include <iostream>
#include <vector>

#include "Utils.h"
#include "CImg.h"

using namespace cimg_library;

void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -f : input image file (default: test.pgm)" << std::endl;
	std::cerr << "  -b : select a bin size" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

int main(int argc, char** argv) {
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;
	string image_filename = "../images/test.ppm";
	unsigned int binnumberstuff = 256;

	for (int i = 1; i < argc; i++) {
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if ((strcmp(argv[i], "-f") == 0) && (i < (argc - 1))) { image_filename = argv[++i]; }
		else if ((strcmp(argv[i], "-b") == 0) && (i < (argc - 1))) { binnumberstuff = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); return 0; }
	}

	cimg::exception_mode(0);

	//detect any potential exceptions
	try {
		//Byte array of image that contains pixel storing for each colour.
		CImg<unsigned char> image_input(image_filename.c_str());
		CImgDisplay disp_input(image_input, "input");

		int maximage = image_input.max();
		//Part 3 - host operations
		//3.1 Select computing devices
		cl::Context context = GetContext(platform_id, device_id);

		//display the selected device
		std::cout << "Runing on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		//create a queue to which we will push commands for the device
		//--Enable queue--//
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);



		//3.2 Load & build the device code
		cl::Program::Sources sources;

		AddSources(sources, "Kernels/kernel.cl");

		cl::Program program(context, sources);

		//build and debug the kernel code
		try {
			program.build();
		}
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}
		//---------------------------------------------------------------------------------------------------------------------------------------------------------//
		
		//Part 4 - device operations
		
		vector<unsigned int> vectorstuff(binnumberstuff);
		size_t histogramsize = vectorstuff.size() * sizeof(unsigned int);

		//--------------------------------------------------------------------------------------------------------------------------------------------------------//
		//device - buffers
		cl::Buffer dev_image_input(context, CL_MEM_READ_ONLY, image_input.size());
		
		cl::Buffer histogrambuffer(context, CL_MEM_READ_WRITE, histogramsize); //tells how many bites it takes up

		cl::Buffer binnum(context, CL_MEM_READ_ONLY, sizeof(unsigned int));
		cl::Buffer binmax(context, CL_MEM_READ_ONLY, sizeof(unsigned int));
		
		cl::Buffer cumhistogrambuffer(context, CL_MEM_READ_WRITE, histogramsize); //tells how many bites it takes up
		
		cl::Buffer normcumhistmbuffer(context, CL_MEM_READ_WRITE, histogramsize);


		cl::Buffer dev_image_output(context, CL_MEM_READ_WRITE, image_input.size()); //should be the same as input image

		//--------------------------------------------------------------------------------------------------------------------------------------------------------//
		//4.1 Copy images to device memory
		queue.enqueueWriteBuffer(dev_image_input, CL_TRUE, 0, image_input.size()*(sizeof(unsigned char)), &image_input.data()[0]);
		queue.enqueueWriteBuffer(binnum, CL_TRUE, 0, sizeof(binnumberstuff), &binnumberstuff);
		queue.enqueueWriteBuffer(binmax, CL_TRUE, 0, sizeof(maximage), &maximage);
		

				//4.2 Setup and execute the kernel (i.e. device code)

		//--Intensitive Histrogram--//
		cl::Kernel hist_local_simple = cl::Kernel(program, "hist_local_simple");
		hist_local_simple.setArg(0, dev_image_input);
		hist_local_simple.setArg(1, histogrambuffer);
		hist_local_simple.setArg(2, cl::Local(binnumberstuff*sizeof(unsigned int)));
		hist_local_simple.setArg(3, binnum);
		hist_local_simple.setArg(4, binmax);

		cl::Event inten_histo_event;

		queue.enqueueNDRangeKernel(hist_local_simple, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange, NULL, &inten_histo_event);
		queue.enqueueReadBuffer(histogrambuffer, CL_TRUE, 0, histogramsize, &vectorstuff[0]);
		//queue.enqueueFillBuffer(cumhistogrambuffer,0,0, histogramsize);
		
		//--------------------------------------------------------------------------------------------------------------------------------------------------------//

		//--cumulative Histrogram--//
		cl::Kernel scan_hs = cl::Kernel(program, "scan_hs");
		scan_hs.setArg(0, histogrambuffer);
		scan_hs.setArg(1, cumhistogrambuffer);
		
		cl::Event cum_histo_event;
		
		vector<unsigned int> cumvectorstuff(binnumberstuff);
		queue.enqueueNDRangeKernel(scan_hs, cl::NullRange, cl::NDRange(binnumberstuff), cl::NullRange, NULL, &cum_histo_event);
		queue.enqueueReadBuffer(cumhistogrambuffer, CL_TRUE, 0, histogramsize, &cumvectorstuff[0]);
		
		//--------------------------------------------------------------------------------------------------------------------------------------------------------//

		//--Normalise Cumulative Histogram--//
		cl::Kernel normalise = cl::Kernel(program, "normalise");
		normalise.setArg(0, cumhistogrambuffer);
		normalise.setArg(1, normcumhistmbuffer);
		normalise.setArg(2, binmax);

		cl::Event norm_cum_histo_event;

		vector<unsigned int> normcumvectorstuff(binnumberstuff);
		queue.enqueueNDRangeKernel(normalise, cl::NullRange, cl::NDRange(binnumberstuff), cl::NullRange, NULL, &norm_cum_histo_event);
		queue.enqueueReadBuffer(normcumhistmbuffer, CL_TRUE, 0, histogramsize, &normcumvectorstuff[0]);

		//--------------------------------------------------------------------------------------------------------------------------------------------------------//
		//--Back-Projection--//
		cl::Kernel back = cl::Kernel(program, "back");
		back.setArg(0, dev_image_input);
		back.setArg(1, normcumhistmbuffer);
		back.setArg(2, dev_image_output);
		back.setArg(3, binnum);
		back.setArg(4, binmax);

		cl::Event back_proj_event;
		vector<unsigned char> output_buffer(image_input.size());

		queue.enqueueNDRangeKernel(back, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange, NULL, &back_proj_event);
		queue.enqueueReadBuffer(dev_image_output, CL_TRUE, 0, output_buffer.size(), &output_buffer.data()[0]);


		//queue.enqueueReadBuffer(cumhistogrambuffer, CL_TRUE, 0, cumvectorstuff.size() * sizeof(unsigned int), &cumvectorstuff.data()[0]);




	

		


		
		//--Create an event and add it to queue command --//
		cl::Event prof_event;

		//queue.enqueueNDRangeKernel(filterr, cl::NullRange,cl::NDRange(image_input.size()), cl::NullRange, NULL, &prof_event);
		
		//4.3 Copy the result from device to host
		//queue.enqueueReadBuffer(dev_image_output, CL_TRUE, 0, output_buffer.size(), &output_buffer.data()[0]);

		/////


		std::cout << vectorstuff << std::endl;

		std::cout << cumvectorstuff << std::endl;

		std::cout << normcumvectorstuff << std::endl;

		CImg<unsigned char> output_image(output_buffer.data(), image_input.width(), image_input.height(), image_input.depth(), image_input.spectrum());
		CImgDisplay disp_output(output_image, "output");

		while (!disp_input.is_closed() && !disp_output.is_closed()
			&& !disp_input.is_keyESC() && !disp_output.is_keyESC()) {
			disp_input.wait(1);
			disp_output.wait(1);
		}

		//--Display kernel execution time --//
		std::cout << "Kernel execution time [ns]:" <<
		prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
		prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;

		//--Detailed breakdown--//
		std::cout << GetFullProfilingInfo(prof_event, ProfilingResolution::PROF_US)
			<< std::endl;

		//--Memory Transfer--//
		//queue.enqueueWriteBuffer(dev_image_output, CL_TRUE, 0, output_buffer.size(), &output_buffer.data()[0], NULL, &prof_event);
		
	}
	catch (const cl::Error& err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}
	catch (CImgException& err) {
		std::cerr << "ERROR: " << err.what() << std::endl;
	}

	return 0;
}
