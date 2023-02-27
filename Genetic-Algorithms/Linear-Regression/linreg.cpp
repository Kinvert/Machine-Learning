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

  int n = 100u;            // How Many Babies
  int generations = 1000u; // Number of Generations
  int i = 0u;              // Initialize Iterator to Zero

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

    // Check Fitness
    for (int j = 0; j < n; j++) {
      children[j].fitness = std::abs(children[j].m - realM) + std::abs(children[j].b - realB);
    }

    // Sort by Fitness
    std::sort(children.begin(), children.end(), [](const Child& a, const Child& b) {
      return a.fitness < b.fitness;
    });

    // Breed Chads (now legally)
    float midM = (children[0].m + children[1].m) / 2.0f;
    float diffM = std::abs(children[0].m - children[1].m);
    float midB = (children[0].b + children[1].b) / 2.0f;
    float diffB = std::abs(children[0].b - children[1].b);
    std::normal_distribution<float> mutateM(midM, midM + diffM);
    std::normal_distribution<float> mutateB(midB, midB + diffB);
    for (int j = 0; j < n; j++) {
      children[j].m = mutateM(gen);
      children[j].b = mutateB(gen);
      children[j].fitness = 0.0f;
    }

    i++;
  }

  for (int j = 0; j < n; j++) {
    children[j].fitness = std::abs(children[j].m - realM) + std::abs(children[j].b - realB);
  }

  std::sort(children.begin(), children.end(), [](const Child& a, const Child&b) {
    return a.fitness < b.fitness;
  });

  for (int j = 0; j < n; j++) {
    std::cout << children[j].fitness << "\n";
  }

  std::cout << "Prediction: M = " << children[0].m << "\nPrediction: B = " << children[0].b << "\n";
  std::cout << "Real:       M = " << realM         << "\nPrediction: B = " << realB         << "\n";

  return 0;
}
