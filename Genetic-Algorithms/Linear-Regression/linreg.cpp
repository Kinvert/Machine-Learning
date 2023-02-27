#include <iostream>

const int LO = -5u;
const int HI = 5u;

int main() {

  // Loop Prep
  int keepBest = 2u;      // How Many to Keep
  int n = 1000u;          // How Many Babies
  float ms[n];            // Slopes
  float bs[n];            // Intercepts
  int generations = 100u; // Number of Generations
  int i = 0u;             // Initialize Iterator to Zero

  // Randomize First Generation
  // https://stackoverflow.com/a/686373
  float r3 = LO + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(HI-LO)));

  // Main Loop
  while (i < generations) {

    // Check Fitness

    // Store Fitness

    // Sort by Fitness

    // Select Chads

    // Eliminate Soyjacks

    // Breed Chads (now legally)

    std::cout << "text\n";
    std::cout << r3 << "\n";
    i++;
  }

  return 0;
}
