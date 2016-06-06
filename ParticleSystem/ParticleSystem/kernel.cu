

#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#include <chrono>

#include "ParticleRenderer.hpp"
#include "ParticleSystem.hpp"

#include "math.cuh"
#include "noise\noise.h"



int main(int argc, char** argv)
{
	srand(time(NULL));

	ParticleRenderer ren;

	glutInit(&argc, argv);

	ren.initGL();
	ren.initSystem();
	ren.begin();

	
}