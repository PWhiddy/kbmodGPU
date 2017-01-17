/*
 * FakeAsteroid.cpp
 *
 *  Created on: Nov 13, 2016
 *      Author: peter
 */

#include "FakeAsteroid.h"

FakeAsteroid::FakeAsteroid(){};

float FakeAsteroid::generateGaussianNoise(float mu, float sigma)
{
	const float epsilon = std::numeric_limits<float>::min();
	const float two_pi = 2.0*3.14159265358979323846;

	static float z0, z1;
	static bool generate;
	generate = !generate;

	if (!generate)
	   return z1 * sigma + mu;

	float u1, u2;

	u1 = rand() * (1.0 / RAND_MAX);
        u2 = rand() * (1.0 / RAND_MAX);	

	while ( u1 <= epsilon )
	{
	   u1 = rand() * (1.0 / RAND_MAX);
	   u2 = rand() * (1.0 / RAND_MAX);
	}

	z0 = sqrt(-2.0 * log(u1)) * cos(two_pi * u2);
	z1 = sqrt(-2.0 * log(u1)) * sin(two_pi * u2);
	return z0 * sigma + mu;
}

void FakeAsteroid::createImage(short *image, int width, int height,
	float xpos, float ypos, psfMatrix psf, float asteroidLevel, float noiseLevel)
{
	//image = new short[width*height];
	//std::default_random_engine generator;
	//std::normal_distribution<double> noise(1000.0, 200.0);

	for (int j=0; j<height; ++j)
	{
		int row = j*width;
		#pragma omp parallel for
		for (int i=0; i<width; ++i)
		{
			image[row+i] = generateGaussianNoise( 700.0, 100.0); //std::max(noise(generator), 0.0)*noiseLevel;
		}
	}

	int xPixel = int(xpos)-psf.dim/2-1;
	int yPixel = int(ypos)-psf.dim/2-1;
	for (int j=0; j<psf.dim; ++j)
	{
		int y = j+yPixel;
		for (int i=0; i<psf.dim; ++i)
		{
			int x = i+xPixel;
			if (x<width && x > 0 && y<height && y>0)
				image[y*width+x] += int(asteroidLevel*psf.kernel[j*psf.dim+i]);
		}
	}
}
