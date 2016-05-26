
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <SFML/Graphics.hpp>
#include <SFML/OpenGL.hpp>
#include <stdio.h>
#include <stdlib.h>

#define EARTH_KG 100

double * h_pos;
double * h_vel;
double * h_acc;
double * h_mass;

double * d_pos;
double * d_vel;
double * d_acc;
double* d_mass;

int numParticles = 1000;


sf::Shader pointProgram;
sf::Shader niceProgram;

sf::RenderTexture buffer;
sf::RenderTexture finalBuffer;

sf::RenderWindow window(sf::VideoMode(1024, 1024), "SFML works!");

inline double random(const double& max, const double& min = 0)
{
	return ((double)static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) * max + min;
}

inline void genParticles()
{
	int particlesGenerated = 0;
	float xDim = 150000;
	float yDim = 150000;
	float zDim = 150000;

	for (int i = 0; i < numParticles; i++)
	{
		int index = particlesGenerated * 3;
		h_pos[index] = random(xDim, -xDim) + xDim / 2;
		h_pos[index + 1] = random(yDim, -yDim) + yDim / 2;
		h_pos[index + 2] = random(zDim, -zDim) + zDim / 2;
		h_vel[index] = random(50, -25);
		h_vel[index + 1] = random(50, -25);
		h_vel[index + 2] = random(12.5, -6);
		h_acc[index] = 0;
		h_acc[index + 1] = 0;
		h_acc[index + 2] = 0;
		h_mass[particlesGenerated] = EARTH_KG;
		particlesGenerated++;
	}
}

inline void drawParticles()
{
	buffer.clear(sf::Color(0, 0, 0, 0));
	finalBuffer.clear(sf::Color(0, 0, 0, 0));
	buffer.resetGLStates();


	glClearColor(0.0, 0.0, 0.0, 0.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glColor4f(1.0, 0.0, 0.0, .4);
	glBegin(GL_POINTS);
	for (int i = 0; i < numParticles; i++)
	{
		int index = i * 3;
		glVertex3f(h_pos[index]/100, h_pos[index + 1]/100, h_pos[index + 2]/100);
	}
	glEnd();

	buffer.pushGLStates();

	sf::Sprite mySprite(buffer.getTexture());
	pointProgram.setParameter("texture", buffer.getTexture());

	buffer.popGLStates();

	finalBuffer.draw(mySprite, &pointProgram);

	mySprite.setTexture(finalBuffer.getTexture());
	niceProgram.setParameter("texture", finalBuffer.getTexture());

	window.draw(mySprite, &niceProgram);
}

int main()
{
	int pointsPerParticleVec = 3;
	size_t size = sizeof(double) * pointsPerParticleVec * numParticles;

	h_pos = (double*)malloc(size);
	h_vel = (double*)malloc(size);
	h_acc = (double*)malloc(size);
	h_mass = (double*)malloc(size / pointsPerParticleVec);

	d_pos = NULL;
	d_vel = NULL;
	d_acc = NULL;
	cudaError_t err = cudaSuccess;
	//allocate space on GPU
	err = cudaMalloc((void **)&d_pos, size);
	err = cudaMalloc((void **)&d_vel, size);
	err = cudaMalloc((void **)&d_acc, size);
	err = cudaMalloc((void **)&d_mass, size / 3);

	genParticles();

	//copy from cpu to GPU


	err = cudaMemcpy(d_pos, h_pos, size, cudaMemcpyHostToDevice);
	err = cudaMemcpy(d_vel, h_vel, size, cudaMemcpyHostToDevice);
	err = cudaMemcpy(d_acc, h_acc, size, cudaMemcpyHostToDevice);
	err = cudaMemcpy(d_mass, h_mass, size / 3, cudaMemcpyHostToDevice);






	pointProgram.loadFromFile("pointVertex.glsl", "pointFragment.glsl");

	niceProgram.loadFromFile("vertex.glsl", "fragment.glsl");

	int width = window.getSize().x;
	int height = window.getSize().y;
	glViewport(0, 0, 1024, 1024);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(-width / 2, width / 2, -height / 2, height / 2, -1000000000, 1000000000);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	//glMatrixMode(GL_PROJECTION);

	//glOrtho(0.0, 1.0, 0.0, 1.0, -1.0, 1.0);

	glLoadIdentity();

	//glEnable(GL_DEPTH_TEST);
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	sf::Event event;
	while (window.pollEvent(event))
	{
		if (event.type == sf::Event::Closed)
		{
			window.close();
		}
				
		//gpuupdate
		window.clear(sf::Color(0, 0, 0, 0));
		
		drawParticles();



		window.display();
	}

	return 0;
}


