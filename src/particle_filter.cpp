/*
 * particle_filter.cpp
 *
 *  Created on: May 25, 2018
 *      Author: Kushgra Pundeer
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include "particle_filter.h"

using namespace std;

// Random Engine declaration
static default_random_engine gen;

#define EPS 0.001

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	// # of Particle.
	num_particles = 100;

	// Initalize weight
	double weight_init = 1.0/ num_particles;

	// Creating Normal Distributions
	normal_distribution<double> N_x(x, std[0]);
	normal_distribution<double> N_y(y, std[1]);
	normal_distribution<double> N_theta(theta, std[2]);

	// Resize particles and weights to fit to the particles desired
	particles.resize(num_particles);
	weights.resize(num_particles);

	for (int i = 0; i < num_particles; i++)
	{
		particles[i].weight = weight_init;
		particles[i].x = N_x(gen);
		particles[i].y = N_y(gen);
		particles[i].theta = N_theta(gen);
		particles[i].id = i;
		
	}

	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	// Normal distributions for sensor noise
	normal_distribution<double> N_x(0, std_pos[0]);
	normal_distribution<double> N_y(0, std_pos[1]);
	normal_distribution<double> N_theta(0, std_pos[2]);

	for(int i = 0; i < num_particles; i++) {

		if (fabs(yaw_rate) < EPS){
			particles[i].x += velocity * delta_t * cos(particles[i].theta);
			particles[i].y += velocity * delta_t * sin(particles[i].theta);
	
		}
		else{
			particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
			particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
			particles[i].theta += yaw_rate * delta_t;
		}

		// Add Noise 
		particles[i].x += N_x(gen);
		particles[i].y += N_y(gen);
		particles[i].theta += N_theta(gen);

	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	
	for (LandmarkObs& obs_: observations) {

		double min_dist = numeric_limits<double>::max();

		for (const LandmarkObs& pred_: predicted){
			double distance = dist(obs_.x, obs_.y, pred_.x, pred_.y);
			if(distance < min_dist) {
				min_dist = distance;
				obs_.id = pred_.id;
			}
		}
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	for (int p = 0; p < num_particles; p++)
	{
		vector<int> assoc;
		vector<double> sense_x;
		vector<double> sense_y;
		vector<LandmarkObs> trans_observation;
		vector<LandmarkObs> predictions;

		for (unsigned int i = 0; i < map_landmarks.landmark_list.size(); i++)
		{
			// Obtain x, y and Id
			double landmark_x_ = map_landmarks.landmark_list[i].x_f;
			double landmark_y_ = map_landmarks.landmark_list[i].y_f;
			int landmark_id_ = map_landmarks.landmark_list[i].id_i;

			if (fabs(landmark_x_ - particles[p].x) <= sensor_range && fabs(landmark_y_ - particles[p].y) <= sensor_range)
			{
				predictions.push_back(LandmarkObs{landmark_id_, landmark_x_, landmark_y_});
			}
		}

		
		for (unsigned int j = 0; j < observations.size(); j++)
		{
			LandmarkObs obs;
			obs = observations[j];

			// Space transformation
			double trans_obs_x = cos(particles[p].theta) * obs.x - sin(particles[p].theta) * obs.y + particles[p].x;
			double trans_obs_y = sin(particles[p].theta) * obs.x + cos(particles[p].theta) * obs.y + particles[p].y;
			trans_observation.push_back(LandmarkObs{obs.id, trans_obs_x, trans_obs_y});
		}

		dataAssociation(predictions, trans_observation);

		particles[p].weight = 1.0;

		for (unsigned int k = 0; k < trans_observation.size(); k++)
		{
			double obs_x, obs_y;
			obs_x = trans_observation[k].x;
			obs_y = trans_observation[k].y;
			double pred_x, pred_y;

			for (unsigned int l = 0; l < predictions.size(); l++)
			{
				if (predictions[l].id == trans_observation[k].id)
				{
					pred_x = predictions[l].x;
					pred_y = predictions[l].y;
				}
			}
			double obs_weight = (1/(2 * M_PI * std_landmark[0] * std_landmark[1])) * exp(-(pow(pred_x - obs_x, 2)/(2 * pow(std_landmark[0], 2))+(pow(pred_y - obs_y, 2)/(2 * pow(std_landmark[1], 2)))));
			particles[p].weight = particles[p].weight * obs_weight;

		}

	}

}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	vector<Particle> new_p;
	vector<double> weights;
	double beta = 0.0;

	for (int p = 0; p < num_particles; p++){
		weights.push_back(particles[p].weight);
	}

	uniform_int_distribution<int> unitintdist(0, num_particles-1);
	auto index = unitintdist(gen);
	double max_weight = *max_element(weights.begin(), weights.end());
	uniform_real_distribution<double> unirealdist(0.0, max_weight);

	for (int i = 0; i < num_particles; i++) {

    	beta += unirealdist(gen) * 2.0;
    	
    	while (beta > weights[index]) {
    		beta -= weights[index];
    		index = (index + 1) % num_particles;
    	}

    	new_p.push_back(particles[index]);
  	}

  	particles = new_p;

}




Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
    return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
