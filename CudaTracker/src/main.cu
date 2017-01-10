/*
 ============================================================================
 Name        :
 Author      : Peter Whidden
 Version     :
 Copyright   :
 Description :
 ============================================================================
 */

#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <cstdlib>
#include <sstream>
#include <ctime>
#include <math.h>
//#include <algorithm>

#include <fitsio.h>
#include "GeneratorPSF.h"
#include "FakeAsteroid.h"


static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

/*
 * A struct to represent a potential trajectory
 */
struct trajectory {
	// Trajectory velocities
	float xVel; 
	float yVel;
	// Likelyhood
	float lh;
	// Origin
	int x; 
	int y;
	// Number of images summed
	int itCount; 
};

/* 
 * For comparing trajectory structs, so that they can be sorted
 */
int compareTrajectory( const void * a, const void * b)
{
        return (int)(1000.0*(((trajectory*)b)->lh - ((trajectory*)a)->lh));
}

/*
 * Device kernel that compares the provided PSF distribution to the distribution
 * around each pixel in the provided image
 */
__global__ void convolvePSF(int width, int height, int imageCount,
		short *image, short *results, float *psf, int psfRad, int psfDim)
{
	// Find bounds of image
	const int x = blockIdx.x*32+threadIdx.x;
	const int y = blockIdx.y*32+threadIdx.y;
	const int minX = max(x-psfRad, 0);
	const int minY = max(y-psfRad, 0);
	const int maxX = min(x+psfRad, width);
	const int maxY = min(y+psfRad, height);
	const int dx = maxX-minX;
	const int dy = maxY-minY;
	if (dx < 1 || dy < 1 ) return;
	// Read Image
	///*__shared__*/ float convArea[13][13]; //convArea[dx][dy];
	int xCorrection = x-psfRad < 0 ? 0 : psfDim-dx;
	int yCorrection = y-psfRad < 0 ? 0 : psfDim-dy;


	float sumDifference = 0.0;
	for (int i=0; i<dx; ++i)
	{
		// #pragma unroll
		for (int j=0; j<dy; ++j)
		{
			sumDifference += float(image[0*width*height+(minX+i)*height+minY+j]) /*convArea[i][j]*/
					 * psf[(i+xCorrection)*psfDim+j+yCorrection];
		}
	}

	results[x*height+y] = int(sumDifference);//*/convArea[psfRad][psfRad]);

}


__global__ void searchImages(int width, int height, int imageCount, short *images, 
			      int trajectoryCount, trajectory *tests, trajectory *results, int edgePadding)
{

	// Get trajectory origin
	int x = blockIdx.x*32+threadIdx.x;
	int y = blockIdx.y*32+threadIdx.y;
	// Give up if any trajectories will hit image edges
	if (x < edgePadding || x + edgePadding > width ||
	    y < edgePadding || y + edgePadding > height) return;

	trajectory best = { .xVel = 0.0, .yVel = 0.0, .lh = 0.0, 
			     .x = x, .y = y, .itCount = trajectoryCount };
	
	for (int t=0; t<trajectoryCount; ++t)
	{
		float xVel = tests[t].xVel;
		float yVel = tests[t].yVel;
		float currentLikelyhood = 0.0;
		for (int i=0; i<imageCount; ++i)
		{
			currentLikelyhood += /*logf(0.0012*/float( images[ i*width*height + 
				(x+int( xVel*float(i)))*height + y + int( yVel*float(i)) ] ); 	
		}
		
		if ( currentLikelyhood > best.lh )
		{
			best.lh = currentLikelyhood;
			best.xVel = xVel;
			best.yVel = yVel;
		}		
	}	
	
	results[ x*height + y ] = best;	

}


int main(int argc, char* argv[])
{

	float psfSigma = argc > 1 ? atof(argv[1]) : 1.0;

	int imageCount = argc > 2 ? atoi(argv[2]) : 1;

	GeneratorPSF *gen = new GeneratorPSF();

	psfMatrix test = gen->createGaussian(psfSigma);

	gen->printPSF(test);

	FakeAsteroid *asteroid = new FakeAsteroid();


	/// Image/FITS Properties ///

	fitsfile *fptr;
	int status;
	long fpixel = 1, naxis = 2, nelements;//, exposure;
	long naxes[2] = { 1024, 1024 }; // X and Y dimensions
	nelements = naxes[0] * naxes[1];
	std::stringstream ss;
	short **pixelArray = new short*[imageCount];

	// Create asteroid images //
	for (int imageIndex=0; imageIndex<imageCount; ++imageIndex)
	{

		/* Initialize the values in the image with noisy astro */

		float kernelNorm = 1.0/test.kernel[test.dim/2*test.dim+test.dim/2];

		pixelArray[imageIndex] = new short[nelements];
		asteroid->createImage(pixelArray[imageIndex], naxes[0], naxes[1],
	 	    1.0*float(imageIndex)+450.0, 0.0*float(imageIndex)+400.0, test, 35.0*kernelNorm, 0.5);

	}

	/*
	// Load real image into first slot
	ss << "../realImg.fits";
	fits_open_data(&fptr, ss.str().c_str(), READONLY, &status);
	fits_report_error(stderr, status);
	ss.str("");
	ss.clear();
	*/

	std::clock_t t1 = std::clock();

	// Process images on GPU //
	short **result = new short*[nelements];
	float *devicePsf;
	short *deviceImageSource;
	short *deviceImageResult;

	dim3 blocks(32,32);
	dim3 threads(32,32);

	// Allocate Device memory
	CUDA_CHECK_RETURN(cudaMalloc((void **)&devicePsf, sizeof(float)*test.dim*test.dim));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&deviceImageSource, sizeof(short)*nelements));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&deviceImageResult, sizeof(short)*nelements));

	CUDA_CHECK_RETURN(cudaMemcpy(devicePsf, test.kernel,
			sizeof(float)*test.dim*test.dim, cudaMemcpyHostToDevice));

	for (int procIndex=0; procIndex<imageCount; ++procIndex)
	{

		result[procIndex] = new short[nelements];
		// Copy image to
		CUDA_CHECK_RETURN(cudaMemcpy(deviceImageSource, pixelArray[procIndex],
				sizeof(short)*nelements, cudaMemcpyHostToDevice));

		convolvePSF<<<blocks, threads>>> (naxes[0], naxes[1], imageCount, deviceImageSource,
				deviceImageResult, devicePsf, test.dim/2, test.dim); //gpuData, size);

		CUDA_CHECK_RETURN(cudaMemcpy(result[procIndex], deviceImageResult,
				sizeof(short)*nelements, cudaMemcpyDeviceToHost));
	}

	CUDA_CHECK_RETURN(cudaFree(devicePsf));
	CUDA_CHECK_RETURN(cudaFree(deviceImageSource));
	CUDA_CHECK_RETURN(cudaFree(deviceImageResult));

	std::clock_t t2 = std::clock();

	std::cout << imageCount << " images, " <<
			1000.0*(t2 - t1)/(double) (CLOCKS_PER_SEC*imageCount) << " ms per image\n";

	
	///////////////// ADDING IMAGE SEARCHING CODE HERE ////////////////

	
	std::clock_t t3 = std::clock();

	// Search images on GPU //
	
		
	// Setup trajectories to test 
	const int anglesCount = 100;
	float angles[anglesCount];
	for (int an=0; an<anglesCount; ++an)
	{
		angles[an] = 6.283185*float(an)/float(anglesCount);
	}
	const int velocitiesCount = 16;
	float velocities [velocitiesCount] = { 0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92,
				  0.94, 0.96, 0.98, 1.00, 1.02, 1.04, 0.06, 1.08, 1.12 }; 
	int trajCount = anglesCount*velocitiesCount;
	trajectory *trajTests = new trajectory[trajCount];
	for (int a=0; a<anglesCount; ++a)
	{
		for (int v=0; v<velocitiesCount; ++v)
		{
			trajTests[a*velocitiesCount+v].xVel = cos(angles[a])*velocities[v];
			trajTests[a*velocitiesCount+v].yVel = sin(angles[a])*velocities[v]; 
		}
	}
	
	//dim3 blocks(32,32);
	//dim3 threads(32,32);

	// Allocate Host memory to store results in
	trajectory* trajResult = new trajectory[nelements];

	// Allocate Device memory 
	trajectory *deviceTests;
	trajectory *deviceSearchResults;
	short *deviceImages;

	CUDA_CHECK_RETURN(cudaMalloc((void **)&deviceTests, sizeof(trajectory)*trajCount));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&deviceImages, sizeof(short)*nelements*imageCount));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&deviceSearchResults, sizeof(trajectory)*nelements));
	

	// Copy trajectories to search
	CUDA_CHECK_RETURN(cudaMemcpy(deviceTests, trajTests,
			sizeof(trajectory)*trajCount, cudaMemcpyHostToDevice));

	// Copy over convolved images one at a time
	for (int i=0; i<imageCount; ++i)
	{
		CUDA_CHECK_RETURN(cudaMemcpy(deviceImages+nelements*i, result[i],
			sizeof(short)*nelements, cudaMemcpyHostToDevice));
	}

	int padding = 2*imageCount+int(psfSigma)+1;

	// Launch Search
	searchImages<<<blocks, threads>>> (naxes[0], naxes[1], imageCount, deviceImages,
				trajCount, deviceTests, deviceSearchResults, padding);

	// Read back results
	CUDA_CHECK_RETURN(cudaMemcpy(trajResult, deviceSearchResults,
				sizeof(trajectory)*nelements, cudaMemcpyDeviceToHost));

	CUDA_CHECK_RETURN(cudaFree(deviceTests));
	CUDA_CHECK_RETURN(cudaFree(deviceSearchResults));
	CUDA_CHECK_RETURN(cudaFree(deviceImages));

	
	// Find most likely trajectories
	qsort( trajResult, nelements, sizeof(trajectory), compareTrajectory);

	for (int i=0; i<15; ++i)
	{
		std::cout << i+1 << ". Likelihood: "  << trajResult[i].lh << " at x: " << trajResult[i].x << ", y: " << trajResult[i].y
                                << "  and velocity x: " << trajResult[i].xVel << ", y: " << trajResult[i].yVel << "\n" ;
	}

	std::clock_t t4 = std::clock();

	std::cout << imageCount << " images, " <<
			1.0*(t4 - t3)/(double) (CLOCKS_PER_SEC) << " seconds to test " << trajCount 
				<< " possible trajectories starting from " << (nelements-padding) << " pixels. " << "\n";



	//////////////// END IMAGE SEARCHING CODE /////////////////


	std::cout << "Writing images to file... ";

	// Write images to file (TODO: encapsulate in method)
	for (int writeIndex=0; writeIndex<imageCount; ++writeIndex)
	{
		/* initialize status before calling fitsio routines */
		status = 0;
		/* Create file name */
		ss << "../toyimages/original/T" << writeIndex+1 << ".fits";
		fits_create_file(&fptr, ss.str().c_str(), &status);
		ss.str("");
		ss.clear();

		/* Create the primary array image (16-bit short integer pixels */
		fits_create_img(fptr, SHORT_IMG, naxis, naxes, &status);

		/* Write the array of integers to the image */
		fits_write_img(fptr, TSHORT, fpixel, nelements, pixelArray[writeIndex], &status);
		fits_close_file(fptr, &status);
		fits_report_error(stderr, status);

		status = 0;
		/* Create file name */
		ss << "../toyimages/convolved/T" << writeIndex+1 << "conv.fits";
		fits_create_file(&fptr, ss.str().c_str(), &status);
		ss.str("");
		ss.clear();

		/* Create the primary array image (16-bit short integer pixels */
		fits_create_img(fptr, SHORT_IMG, naxis, naxes, &status);

		/* Write the array of integers to the image */
		fits_write_img(fptr, TSHORT, fpixel, nelements, result[writeIndex], &status);
		fits_close_file(fptr, &status);
		fits_report_error(stderr, status);

	}

	std::cout << "Done.\n";

	// Finished!

	/* Free memory */
	for (int im=0; im<imageCount; ++im)
	{
		delete[] pixelArray[im];
		delete[] result[im];
	}

	delete[] pixelArray;
	delete[] result;
	
	delete[] trajResult;
	
	delete gen;
	delete asteroid;

	return 0;
} 

/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err)
{
	if (err == cudaSuccess)
		return;
	std::cerr << statement<<" returned " << cudaGetErrorString(err)
			<< "("<<err<< ") at "<<file<<":"<<line << std::endl;
	exit (1);
}

