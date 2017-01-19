/*
 ============================================================================
 Name        : KBMOD CUDA
 Author      : Peter Whidden
 Description :
 ============================================================================
 */

#include <iostream>
#include <fstream>
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

void writeFitsImg(fitsfile *f, const char *name, long fpix, 
	long *naxes, long nelements, void *array);

const char* parseLine(std::ifstream& cFile, int debug);

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
        return (int)(5000.0*(((trajectory*)b)->lh - ((trajectory*)a)->lh));
}

/*
 * Device kernel that compares the provided PSF distribution to the distribution
 * around each pixel in the provided image
 */
__global__ void convolvePSF(int width, int height, int imageCount,
	float *image, float *results, float *psf, int psfRad, 
	int psfDim, float background, float normalization)
{
	// Find bounds of convolution area
	const int x = blockIdx.x*32+threadIdx.x;
	const int y = blockIdx.y*32+threadIdx.y;
	const int minX = max(x-psfRad, 0);
	const int minY = max(y-psfRad, 0);
	const int maxX = min(x+psfRad, width-1);
	const int maxY = min(y+psfRad, height-1);
	const int dx = maxX-minX;
	const int dy = maxY-minY;
	if (dx < 1 || dy < 1 ) return;
 
	// Read kernel
	float sumDifference = 0.0;
	for (int j=minY; j<=maxY; ++j)
	{
		// #pragma unroll
		for (int i=minX; i<=maxX; ++i)
		{
			sumDifference += (image[j*width+i] - background)
					 * psf[(j-minY)*psfDim+i-minX];
		}
	}

	results[y*width+x] = sumDifference*normalization;

}

/*
 * Searches through images (represented as a flat array of floats) looking for most likely
 * trajectories in the given list. Outputs a results image of best trajectories. Note that
 * for now only the single best trajectory starting at each pixel makes it to results. 
 */
__global__ void searchImages(int width, int height, int imageCount, float *images, 
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
			currentLikelyhood += logf( images[ i*width*height + 
				(y+int( yVel*float(i)))*width +
				 x + int( xVel*float(i)) ] ); 	
		}
		
		if ( currentLikelyhood > best.lh )
		{
			best.lh = currentLikelyhood;
			best.xVel = xVel;
			best.yVel = yVel;
		}		
	}	
	
	results[ y*width + x ] = best;	

}


int main(int argc, char* argv[])
{

	/* Read parameters from config file */
	std::ifstream pFile ("parameters.config");
    	if (!pFile.is_open()) 
		std::cout << "Unable to open parameters file." << '\n';
	
	int debug             = atoi(parseLine(pFile, false));
	int imageCount        = atoi(parseLine(pFile, debug));
	float psfSigma        = atof(parseLine(pFile, debug));
	float asteroidLevel   = atof(parseLine(pFile, debug));
	float initialX        = atof(parseLine(pFile, debug));
	float initialY        = atof(parseLine(pFile, debug));
	float velocityX       = atof(parseLine(pFile, debug));
	float velocityY       = atof(parseLine(pFile, debug));
	float backgroundLevel = atof(parseLine(pFile, debug));
	float backgroundSigma = atof(parseLine(pFile, debug));
	int anglesCount       = atoi(parseLine(pFile, debug));
	int velocitySteps     = atoi(parseLine(pFile, debug));
	float minVelocity     = atof(parseLine(pFile, debug));
	float maxVelocity     = atof(parseLine(pFile, debug));
	int writeFiles        = atoi(parseLine(pFile, debug));
	std::string origPath  = parseLine(pFile, debug);
	std::string psiPath   = parseLine(pFile, debug);
	pFile.close();
     
	/* Create instances of psf and object generators */
	GeneratorPSF *gen = new GeneratorPSF();

	psfMatrix testPSF = gen->createGaussian(psfSigma);

	float psfCoverage = gen->printPSF(testPSF, debug);

	FakeAsteroid *asteroid = new FakeAsteroid();


	/* Setup Image/FITS Properties of test Images  */
	fitsfile *fptr;
	long fpixel = 1, /*naxis = 2,*/ nelements;//, exposure;
	long naxes[2] = { 1024, 1024 }; // X and Y dimensions
	nelements = naxes[0] * naxes[1];
	std::stringstream ss;
	float **pixelArray = new float*[imageCount];

	// Create asteroid images //
	for (int imageIndex=0; imageIndex<imageCount; ++imageIndex)
	{

		/* Initialize the values in the image with noisy astro */

		pixelArray[imageIndex] = new float[nelements];
		asteroid->createImage( pixelArray[imageIndex], naxes[0], naxes[1],
	 	    	velocityX*float(imageIndex)+initialX,  // Asteroid X position 
			velocityY*float(imageIndex)+initialY, // Asteroid Y position
			testPSF, asteroidLevel, backgroundLevel, backgroundSigma);

	}

	/*
	// Load real image into first slot
	ss << "../realImg.fits";
	fits_open_data(&fptr, ss.str().c_str(), READONLY, &status);
	fits_report_error(stderr, status);
	ss.str("");
	ss.clear();
	*/


	/* Generate psi images on device */

	std::clock_t t1 = std::clock();

	// Pointers to device memory //
	float **result = new float*[nelements];
	float *devicePsf;
	float *deviceImageSource;
	float *deviceImageResult;

	dim3 blocks(32,32);
	dim3 threads(32,32);

	// Allocate Device memory
	CUDA_CHECK_RETURN(cudaMalloc((void **)&devicePsf, sizeof(float)*testPSF.dim*testPSF.dim));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&deviceImageSource, sizeof(float)*nelements));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&deviceImageResult, sizeof(float)*nelements));

	CUDA_CHECK_RETURN(cudaMemcpy(devicePsf, testPSF.kernel,
		sizeof(float)*testPSF.dim*testPSF.dim, cudaMemcpyHostToDevice));

	for (int procIndex=0; procIndex<imageCount; ++procIndex)
	{

		result[procIndex] = new float[nelements];
		// Copy image to
		CUDA_CHECK_RETURN(cudaMemcpy(deviceImageSource, pixelArray[procIndex],
			sizeof(float)*nelements, cudaMemcpyHostToDevice));

		convolvePSF<<<blocks, threads>>> (naxes[0], naxes[1], 
			imageCount, deviceImageSource, deviceImageResult, devicePsf, 
			testPSF.dim/2, testPSF.dim, backgroundLevel, 1.0/psfCoverage);

		CUDA_CHECK_RETURN(cudaMemcpy(result[procIndex], deviceImageResult,
			sizeof(float)*nelements, cudaMemcpyDeviceToHost));
	}

	CUDA_CHECK_RETURN(cudaFree(devicePsf));
	CUDA_CHECK_RETURN(cudaFree(deviceImageSource));
	CUDA_CHECK_RETURN(cudaFree(deviceImageResult));

	std::clock_t t2 = std::clock();

	std::cout << imageCount << " images, " <<
		1000.0*(t2 - t1)/(double) (CLOCKS_PER_SEC*imageCount) 
		  << " ms per image\n";


	///* Search images on GPU *///
	
	std::clock_t t3 = std::clock();
		
	/* Create test trajectories */
	float *angles = new float[anglesCount];
	for (int i=0; i<anglesCount; ++i)
	{
		angles[i] = 6.283185*float(i)/float(anglesCount);
	}

	float *velocities = new float[velocitySteps];
	float dv = (maxVelocity-minVelocity)/float(velocitySteps);
	for (int i=0; i<velocitySteps; ++i)
	{
		velocities[i] = minVelocity+float(i)*dv;	
	}	
 
	int trajCount = anglesCount*velocitySteps;
	trajectory *trajTests = new trajectory[trajCount];
	for (int a=0; a<anglesCount; ++a)
	{
		for (int v=0; v<velocitySteps; ++v)
		{
			trajTests[a*velocitySteps+v].xVel = cos(angles[a])*velocities[v];
			trajTests[a*velocitySteps+v].yVel = sin(angles[a])*velocities[v]; 
		}
	}

	// Allocate Host memory to store results in
	trajectory* trajResult = new trajectory[nelements];

	// Allocate Device memory 
	trajectory *deviceTests;
	trajectory *deviceSearchResults;
	float *deviceImages;

	CUDA_CHECK_RETURN(cudaMalloc((void **)&deviceTests, sizeof(trajectory)*trajCount));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&deviceImages, sizeof(float)*nelements*imageCount));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&deviceSearchResults, sizeof(trajectory)*nelements));
	

	// Copy trajectories to search
	CUDA_CHECK_RETURN(cudaMemcpy(deviceTests, trajTests,
			sizeof(trajectory)*trajCount, cudaMemcpyHostToDevice));

	// Copy over psi images one at a time
	for (int i=0; i<imageCount; ++i)
	{
		CUDA_CHECK_RETURN(cudaMemcpy(deviceImages+nelements*i, result[i],
			sizeof(float)*nelements, cudaMemcpyHostToDevice));
	}

	// assumes object is not moving more than 2 pixels per image
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

	
	// Sort results by likelihood
	qsort(trajResult, nelements, sizeof(trajectory), compareTrajectory);
	if (debug)
	{
		for (int i=0; i<15; ++i)
		{
			if (i+1 < 10) std::cout << " ";
			std::cout << i+1 << ". Likelihood: "  << trajResult[i].lh 
				  << " at x: " << trajResult[i].x << ", y: " << trajResult[i].y
				  << "  and velocity x: " << trajResult[i].xVel 
				  << ", y: " << trajResult[i].yVel << "\n" ;
		}
	}

	std::clock_t t4 = std::clock();

	std::cout << imageCount << " images, " <<
		1.0*(t4 - t3)/(double) (CLOCKS_PER_SEC) << " seconds to test " 
		<< trajCount << " possible trajectories starting from " 
		<< ((naxes[0]-padding)*(naxes[1]-padding)) << " pixels. " << "\n";

	std::cout << "Writing images to file... ";

	// Write images to file 
	if (writeFiles)
	{
		for (int writeIndex=0; writeIndex<imageCount; ++writeIndex)
		{
			/* Create file name */
			ss << origPath << "T";
			// Add leading zeros to filename
			if (writeIndex+1<100) ss << "0";
			if (writeIndex+1<10) ss << "0";
			ss << writeIndex+1 << ".fits";
			writeFitsImg(fptr, ss.str().c_str(), fpixel, naxes, 
				nelements, pixelArray[writeIndex]);
			ss.str("");
			ss.clear();		

			ss << psiPath << "T";
			if (writeIndex+1<100) ss << "0";
			if (writeIndex+1<10) ss << "0"; 
			ss << writeIndex+1 << "psi.fits";
			writeFitsImg(fptr, ss.str().c_str(), fpixel, naxes, 
				nelements, pixelArray[writeIndex]);
			ss.str("");
			ss.clear();
		}
	}
	std::cout << "Done.\n";

	/* Write results file */
	// std::cout needs to be rerouted to output to console after this...
	std::freopen("results.txt", "w", stdout);
	std::cout << "# t0_x t0_y theta_par theta_perp v_x v_y likelihood est_flux\n";
        for (int i=0; i<20; ++i)
        {
                std::cout << trajResult[i].x << " " << trajResult[i].y << " 0.0 0.0 "
                          << trajResult[i].xVel << " " << trajResult[i].yVel << " "       
                          << trajResult[i].lh << " 0.0\n" ;
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
	
	delete[] angles;
	delete[] velocities;
	delete[] trajTests;	
	delete[] trajResult;
	
	delete gen;
	delete asteroid;

	return 0;
} 

const char* parseLine(std::ifstream& pFile, int debug)
{
	std::string line;
	getline(pFile, line);
        int delimiterPos = line.find(":");
	if (debug) 
	{
		std::cout << line.substr(0, delimiterPos );
		std::cout << " : " << line.substr(delimiterPos + 2) << "\n";
	}
	return (line.substr(delimiterPos + 2)).c_str();
}

void writeFitsImg(fitsfile *f, const char *name, long fpix, long *naxes, long nelements, void *array)
{
	/* initialize status before calling fitsio routines */
	int status = 0;
        /* Create file with name */
	fits_create_file(&f, name, &status);

	/* Create the primary array image (32-bit float pixels */
	fits_create_img(f, FLOAT_IMG, 2 /*naxis*/, naxes, &status);

	/* Write the array of floats to the image */
	fits_write_img(f, TFLOAT, 1, nelements, array, &status);
	fits_close_file(f, &status);
	fits_report_error(stderr, status);
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

