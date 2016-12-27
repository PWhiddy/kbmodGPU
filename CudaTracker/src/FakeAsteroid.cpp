/*
 * FakeAsteroid.cpp
 *
 *  Created on: Nov 13, 2016
 *      Author: peter
 */

#include "FakeAsteroid.h"

FakeAsteroid::FakeAsteroid(){};

void FakeAsteroid::createImage(short *image, int width, int height,
	float xpos, float ypos, psfMatrix psf, float asteroidLevel, float noiseLevel)
{
	//image = new short[width*height];
	std::default_random_engine generator;
	std::normal_distribution<double> noise(1000.0, 200.0);

	for (int i=0; i<height; ++i)
	{
		int row = i*width;
		#pragma omp parallel for
		for (int j=0; j<width; ++j)
		{
			image[row+j] = std::max(noise(generator), 0.0)*noiseLevel;
		}
	}

	int xPixel = int(xpos * width);
	int yPixel = int(ypos * height);
	for (int i=0; i<psf.dim; ++i)
	{
		int x = xPixel+i;
		for (int j=0; j<psf.dim; ++j)
		{
			int y = yPixel+j;
			if (x<width && y<height)
				image[x*width+y] += asteroidLevel*psf.kernel[i*psf.dim+j];
		}
	}
}
