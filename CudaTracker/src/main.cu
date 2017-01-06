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

	/*
	// float sum = 0.0;
	for (int i=0; i<dx; ++i)
	{
		#pragma unroll
		for (int j=0; j<dy; ++j)
		{
		//	float value = float(image[0*width*height+(minX+i)*height+minY+j]);
		//	sum += value;
			convArea[i][j] = float(image[0*width*height+(minX+i)*height+minY+j]);//value;
		}
	}

	float sumDifference = 0.0;
	for (int i=0; i<dx; ++i)
	{
		#pragma unroll
		for (int j=0; j<dy; ++j)
		{
			sumDifference += convArea[i][j]/* /sum* / * psf[(i+xCorrection)*psfDim+j+yCorrection];
		}
	}
	*/

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

	results[0*width*height+x*height+y] = int(sumDifference);//*/convArea[psfRad][psfRad]);

}


__global__ void searchImages(int width, int height, int imageCount, short *images, 
			      int trajectoryCount, trajectory *tests, trajectory *results, int edgePadding)
{

	// Get trajectory origin
	int x = blockIdx.x*32+threadIdx.x;
	int y = blockIdx.y*32+threadIdx.y;
	// Give up if any trajectories will hit image edges
	if (x < edgePadding || x + edgePadding > width || y < edgePadding || y + edgePadding > height) return;

	trajectory best = { .xVel = 0.0, .yVel = 0.0, .lh = 0.0, .x = x, .y = y, .itCount = trajectoryCount };
	
	for (int t=0; t<trajectoryCount; ++t)
	{
		float xVel = tests[t].xVel;
		float yVel = tests[t].yVel;
		float currentLikelyhood = 0.0;
		for (int i=0; i<imageCount; ++i)
		{
			currentLikelyhood += float( images[ i*width*height + (x+int( xVel*float(i)))*height + y + int( yVel*float(i)) ] ); 	
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
	 	    0.000952*float(imageIndex)+0.25, 0.000885*float(imageIndex)+0.2, test, 450.0*kernelNorm, 0.5);

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

	
	std::clock_t t1 = std::clock();

	// Search images on GPU //
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




	//////////////// END IMAGE SEARCHING CODE /////////////////


	// Write images to file (TODO: encapsulate in method)
	for (int writeIndex=0; writeIndex<imageCount; ++writeIndex)
	{
		/* initialize status before calling fitsio routines */
		status = 0;
		/* Create file name */
		ss << "Asteroid" << writeIndex+1 << ".fits";
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
		ss << "AsteroidPSFConv" << writeIndex+1 << ".fits";
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

	// Finished!

	/* Free memory */
	for (int im=0; im<imageCount; ++im)
	{
		delete[] pixelArray[im];
		delete[] result[im];
	}

	delete[] pixelArray;
	delete[] result;
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

