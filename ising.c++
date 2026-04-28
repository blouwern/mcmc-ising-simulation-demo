#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

// simulation constants
constexpr int L = 6;
constexpr int N = L * L;
constexpr double J = 1.0;
constexpr double kB = 1.0;
constexpr int EQ_STEPS = 20000;
constexpr int MC_STEPS = 100000;

// RNG utilities
std::mt19937 rng(std::random_device{}());
std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);
std::uniform_int_distribution<int> site_dist(0, N - 1);

class Lattice {
public:
    std::vector<int> spin;
    double energy;
    double magnet;

    Lattice() : spin(N, 1), energy(0.0), magnet(N) {
        std::uniform_int_distribution<int> spin_dist(0, 1);
        for (int i = 0; i < N; ++i) {
            spin[i] = spin_dist(rng) == 0 ? -1 : 1;
        }
        magnet = std::accumulate(spin.begin(), spin.end(), 0.0);
        energy = total_energy();
    }

    int idx(int x, int y) const {
        x = (x + L) % L;
        y = (y + L) % L;
        return y * L + x;
    }

    double total_energy() const {
        double E = 0.0;
        for (int y = 0; y < L; ++y) {
            for (int x = 0; x < L; ++x) {
                int s = spin[idx(x, y)];
                E += -J * s * spin[idx(x + 1, y)];
                E += -J * s * spin[idx(x, y + 1)];
            }
        }
        return E;
    }

    void flip(int i, double dE) {
        spin[i] *= -1;
        energy += dE;
        magnet += 2.0 * spin[i];
    }

    double delta_energy(int i) const {
        int x = i % L;
        int y = i / L;
        int s = spin[i];
        int nb_sum = spin[idx(x + 1, y)] + spin[idx(x - 1, y)] + spin[idx(x, y + 1)] + spin[idx(x, y - 1)];
        return 2.0 * J * s * nb_sum;
    }
};

// Local Metropolis update
void metropolis_sweep(Lattice& lat, double beta) {
    for (int k = 0; k < N; ++k) {
        int i = site_dist(rng);
        double dE = lat.delta_energy(i);
        if (dE <= 0.0 || uniform_dist(rng) < std::exp(-beta * dE)) {
            lat.flip(i, dE);
        }
    }
}

// Wolff cluster update
void wolff_update(Lattice& lat, double beta) {
    int seed = site_dist(rng);
    std::vector<bool> in_cluster(N, false);
    std::vector<int> stack = {seed};
    in_cluster[seed] = true;

    double p_add = 1.0 - std::exp(-2.0 * beta * J);

    while (!stack.empty()) {
        int i = stack.back();
        stack.pop_back();

        int x = i % L;
        int y = i / L;
        int s = lat.spin[i];

        static const int dx[] = {1, -1, 0, 0};
        static const int dy[] = {0, 0, 1, -1};
        for (int dir = 0; dir < 4; ++dir) {
            int nx = (x + dx[dir] + L) % L;
            int ny = (y + dy[dir] + L) % L;
            int j = ny * L + nx;
            if (lat.spin[j] == s && !in_cluster[j] && uniform_dist(rng) < p_add) {
                in_cluster[j] = true;
                stack.push_back(j);
            }
        }
    }

    double dE_total = 0.0;
    for (int i = 0; i < N; ++i) {
        if (!in_cluster[i]) {
            continue;
        }
        int x = i % L;
        int y = i / L;
        int s = lat.spin[i];
        for (int dir = 0; dir < 4; ++dir) {
            int nx = (x + (dir == 0 ? 1 : (dir == 1 ? -1 : 0)) + L) % L;
            int ny = (y + (dir == 2 ? 1 : (dir == 3 ? -1 : 0)) + L) % L;
            int j = ny * L + nx;
            if (!in_cluster[j]) {
                dE_total += 2.0 * J * s * lat.spin[j];
            }
        }
    }

    for (int i = 0; i < N; ++i) {
        if (in_cluster[i]) {
            lat.spin[i] *= -1;
            lat.magnet += 2.0 * lat.spin[i];
        }
    }
    lat.energy += dE_total;
}

struct Results {
    double E_avg;
    double M_avg;
    double Cv;
    double chi;
};

template <typename UpdateFunc>
Results simulate(double T, UpdateFunc update, bool is_wolff = false) {
    Lattice lat;
    double beta = 1.0 / (kB * T);

    int eq_steps = is_wolff ? EQ_STEPS : EQ_STEPS;
    for (int t = 0; t < eq_steps; ++t) {
        update(lat, beta);
    }

    double sum_E = 0.0;
    double sum_E2 = 0.0;
    double sum_M = 0.0;
    double sum_M2 = 0.0;
    int measure_steps = MC_STEPS;

    for (int t = 0; t < measure_steps; ++t) {
        update(lat, beta);
        double E = lat.energy;
        double M = lat.magnet;
        double absM = std::abs(M);
        sum_E += E;
        sum_E2 += E * E;
        sum_M += absM;
        sum_M2 += M * M;
    }

    double norm = 1.0 / measure_steps;
    double avg_E = sum_E * norm / N;
    double avg_E_total = sum_E * norm;
    double avg_E2_total = sum_E2 * norm;
    double avg_M_abs = sum_M * norm / N;
    double avg_M2_total = sum_M2 * norm;

    double Cv = (beta * beta / N) * (avg_E2_total - avg_E_total * avg_E_total);
    double chi = (beta / N) * avg_M2_total;

    return {avg_E, avg_M_abs, Cv, chi};
}

template <typename UpdateFunc>
Results simulate_rigorous(double T, UpdateFunc update, bool is_wolff = false) {
    Lattice lat;
    double beta = 1.0 / T;

    for (int t = 0; t < EQ_STEPS; ++t) {
        update(lat, beta);
    }

    double sum_E = 0.0;
    double sum_E2 = 0.0;
    double sum_M = 0.0;
    double sum_M2 = 0.0;
    for (int t = 0; t < MC_STEPS; ++t) {
        update(lat, beta);
        double E = lat.energy;
        double M = lat.magnet;
        sum_E += E;
        sum_E2 += E * E;
        sum_M += M;
        sum_M2 += M * M;
    }

    double norm = 1.0 / MC_STEPS;
    double avg_E_total = sum_E * norm;
    double avg_E2_total = sum_E2 * norm;
    double avg_M_total = sum_M * norm;
    double avg_M2_total = sum_M2 * norm;

    double avg_E_per = avg_E_total / N;
    return {
        avg_E_per,
        0.0,
        (beta * beta / N) * (avg_E2_total - avg_E_total * avg_E_total),
        (beta / N) * (avg_M2_total - avg_M_total * avg_M_total),
    };
}

template <typename UpdateFunc>
Results simulate_full(double T, UpdateFunc update) {
    Lattice lat;
    double beta = 1.0 / T;

    for (int t = 0; t < EQ_STEPS; ++t) {
        update(lat, beta);
    }

    double sum_E = 0.0;
    double sum_E2 = 0.0;
    double sum_M_abs = 0.0;
    double sum_M2 = 0.0;
    double sum_M = 0.0;
    for (int t = 0; t < MC_STEPS; ++t) {
        update(lat, beta);
        double E = lat.energy;
        double M = lat.magnet;
        sum_E += E;
        sum_E2 += E * E;
        sum_M += M;
        sum_M_abs += std::fabs(M);
        sum_M2 += M * M;
    }

    double n = MC_STEPS;
    double avg_E = sum_E / n / N;
    double avg_M_abs = sum_M_abs / n / N;
    double avg_E2 = sum_E2 / n;
    double avg_E_ = sum_E / n;
    double avg_M2 = sum_M2 / n;
    double avg_M_ = sum_M / n;
    double Cv = (beta * beta / N) * (avg_E2 - avg_E_ * avg_E_);
    double chi = (beta / N) * (avg_M2 - avg_M_ * avg_M_);
    return {avg_E, avg_M_abs, Cv, chi};
}

int main() {
    std::vector<double> temps;
    for (double T = 0.0; T <= 10.0; T += 0.1) {
        temps.push_back(T);
    }

    std::ofstream fout_metro("ising_metro_6x6.csv");
    std::ofstream fout_wolff("ising_wolff_6x6.csv");
    fout_metro << "T,E_per_spin,M_abs_per_spin,Cv_per_spin,chi_per_spin\n";
    fout_wolff << "T,E_per_spin,M_abs_per_spin,Cv_per_spin,chi_per_spin\n";

    std::cout << "Simulating Metropolis...\n";
    for (double T : temps) {
        Results res = simulate_full(T, metropolis_sweep);
        fout_metro << T << "," << res.E_avg << "," << res.M_avg << "," << res.Cv << "," << res.chi << "\n";
        std::cout << "Metro T=" << T << " done.\n";
    }

    std::cout << "Simulating Wolff...\n";
    for (double T : temps) {
        Results res = simulate_full(T, [](Lattice& lat, double beta) { wolff_update(lat, beta); });
        fout_wolff << T << "," << res.E_avg << "," << res.M_avg << "," << res.Cv << "," << res.chi << "\n";
        std::cout << "Wolff T=" << T << " done.\n";
    }

    fout_metro.close();
    fout_wolff.close();
    std::cout << "Results saved to ising_metro_6x6.csv and ising_wolff_6x6.csv\n";
    return 0;
}
