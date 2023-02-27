#include <algorithm>
#include <iostream>
#include <random>
#include <vector>

const float LO = -5.0f;
const float HI = 5.0f;

const float realM = 1.0f;
const float realB = -1.0f;

struct Child {
  float fitness;
  float m;
  float b;
};

int main() {

  // Loop Prep
  int keepBest = 2u;      // How Many to Keep
  int n = 1000u;          // How Many Babies
  float ms[n];            // Slopes
  float bs[n];            // Intercepts
  int generations = 100u; // Number of Generations
  int i = 0u;             // Initialize Iterator to Zero

  // Randomize First Generation
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(LO, HI);

  std::vector<Child> children(n);
  for (int i = 0; i < n; i++) {
    children[i].fitness = 0.0f;
    children[i].m = dist(gen);
    children[i].b = dist(gen);
  }

  // Main Loop
  while (i < generations) {

    // For Child in Children
    for (int j = 0; j < n; j++) {
      // Check Fitness
      children[j].fitness = std::abs(children[j].m - realM) + std::abs(children[j].b - realB);
    }

    // Sort by Fitness
    std::sort(children.begin(), children.end(), [](const Child& a, const Child& b) {
        return a.fitness < b.fitness;
    });

    // Select Chads

    // Eliminate Soyjacks

    // Breed Chads (now legally)

    i++;
  }

  for (int j = 0; j < n; j++) {
    std::cout << children[j].fitness << "\n";
  }

  return 0;
}
