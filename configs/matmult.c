#define N 1024

float a[N][N], b[N][N], c[N][N];

void initialize() {
    int i, j;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            c[i][j] = 0.0;
            a[i][j] = ((i + j) % 3) - 1;
            b[i][j] = ((i * j) % 3) - 1;
        }
    }
}

#pragma rstream map
void matmult() {
    int i, j, k;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            c[i][j] = 0.0;
            for (k = 0; k < N; k++) {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
}

int main() {
    initialize();
    matmult();
    return 0;
}