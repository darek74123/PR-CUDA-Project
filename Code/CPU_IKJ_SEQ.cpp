#include <chrono>
#include <fstream>
#include <iostream>
#include <string>

const int SIZE = 864;

float matrix_a[SIZE][SIZE];
float matrix_b[SIZE][SIZE];
float matrix_r[SIZE][SIZE];

void initialize_matrices()
{
    for (int i = 0; i < SIZE; i++)
    {
        for (int j = 0; j < SIZE; j++)
        {
            matrix_a[i][j] = 1.0f;
            matrix_b[i][j] = 0.01f;
            matrix_r[i][j] = 0.0f;
        }
    }
}

void multiply_matrices_IKJ_sequential()
{
    for (int i = 0; i < SIZE; i++)
        for (int k = 0; k < SIZE; k++)
            for (int j = 0; j < SIZE; j++)
                matrix_r[i][j] += matrix_a[i][k] * matrix_b[k][j];
}

int main()
{
    std::ofstream file;
    file.open("results.txt", std::ofstream::out | std::ofstream::app);

    std::cout << "Matrix size: " << SIZE << std::endl;
    file << "Matrix size: " << SIZE << std::endl;

    initialize_matrices();
    {
        auto start_chrono = std::chrono::high_resolution_clock::now();

        multiply_matrices_IKJ_sequential();

        auto stop_chrono = std::chrono::high_resolution_clock::now();
        auto time_span = std::chrono::duration_cast<std::chrono::duration<double>>(stop_chrono - start_chrono);

        std::cout << "IKJ_seq:     " << time_span.count() << " seconds." << std::endl;
        file << "IKJ_seq:     " << time_span.count() << " seconds." << std::endl;
    }

    file << "\n---------------------------------------------------------------\n\n";
    file.close();

    return 0;
}
