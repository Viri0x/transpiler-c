#include <stdio.h>
/* Predicting linear regression from trained model. */

    /* Utils */
    float fact(int x, int n_fact)
    {
        float res = 1.0;
        
        for (int i = 1; i <= n_fact; i++)
        {
            res *= i;
        }
        
        return res;
    }

    float pow_(int x, int n_pow)
    {
        float res = 1.0;
        
        for (int i = 0; i < n_pow; i++)
        {
            res *= x;
        }
        
        return res;
    }

    float exp_approx(float x, int n_term)
    {
        float res = 1.0;
        
        for (int i = 1; i <= n_term; i++)
        {
            res += pow_(x, i) / fact(x, i);
        }
        
        return res;
    }

    float sigmoid(float x)
    {
        return 1.0 / (1.0 + exp_approx(-x, 10));
    }

    /* Code full logistic regression*/

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

    float logistic_regression(float *features, float* theta, int n_parameter)
    {
        float pred = sigmoid(linear_regression_prediction(features, theta, n_parameter));
        
        if (pred <= 0.5)
            return 0.0;
        else
            return 1.0;
    }

    int main (int argc, char *arvgv[])
    {
        float features_arr[2] = {3.141592, 2.714836};
        float coef_arr[3] = {0.22373607697263656,1.1835153640603044, -2.687063705810736};
        float *features = features_arr;
        float *coefs = coef_arr;
        int n_coefs = 3;

        /* Logistic regression pred */
        printf("Prediction: %f", logistic_regression(features, coefs, n_coefs));

        return 0;
    }