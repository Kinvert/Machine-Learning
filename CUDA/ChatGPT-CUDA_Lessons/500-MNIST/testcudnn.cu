#include <cudnn.h>
#include <stdio.h>

int main() {
    int version = cudnnGetVersion();
    printf("cudnn library version: %d\n", version);
    return 0;
}
