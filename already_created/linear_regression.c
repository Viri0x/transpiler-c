#include <stdio.h>
/* Predicting linear regression from trained model. */

    float linear_regression_prediction(float* features, float* thetas, int n_thetas)
    {
        float pred = thetas[0];
        
        for (int i = 1; i < n_thetas; i++)
        {
            /* Multiply thetas with features and sum ?*/
            pred += features[i - 1] * thetas[i];
        }
            
        return pred;
    }
        
    int main (int argc, char *arvgv[])
    {
        float features_arr[2] = {3.141592, 2.714836};
        float coef_arr[3] = {-0.03059269547364596,1.8391381815890269, -0.5546497522328709};
        float *features = features_arr;
        float *coefs = coef_arr;
        int n_coefs = 3;

        /* Linear regression pred */
        printf("Prediction: %f", linear_regression_prediction(features, coefs, n_coefs));

        return 0;
    }