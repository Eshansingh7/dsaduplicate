#include <stdio.h>
#include <stdlib.h>

int* findDuplicates(int* nums, int numsSize, int* returnSize) {
    int* duplicates = (int*)malloc(numsSize * sizeof(int));
    int index = 0;

    for (int i = 0; i < numsSize; i++) {
        int num = abs(nums[i]);

        if (nums[num - 1] < 0) {
            duplicates[index++] = num;
        } else {
            nums[num - 1] = -nums[num - 1];
        }
    }

    *returnSize = index;
    return duplicates;
}

int main() {
    int nums[] = {4, 3, 2, 7, 8, 2, 3, 1};
    int numsSize = sizeof(nums) / sizeof(nums[0]);

    int returnSize;
    int* duplicates = findDuplicates(nums, numsSize, &returnSize);

    printf("Duplicates: ");
    for (int i = 0; i < returnSize; i++) {
        printf("%d ", duplicates[i]);
    }
    printf("\n");

    free(duplicates);

    return 0;
}
