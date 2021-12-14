import joblib
import math
import subprocess
import numpy as np

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def get_coefs_as_c(model, type):

    # get coefs and bias from model
    bias = model.intercept_
    coefs = model.coef_

    if type == "logistic_regression":
        bias = bias[0]
        coefs = coefs[0]

    # Create string from coefs for c
    coefs_string = "{" + str(bias) + ","

    for i in range(len(coefs)):
        coefs_string += str(coefs[i])
        if i < len(coefs) - 1:
            coefs_string += ", "
    coefs_string += "}"

    return coefs_string, len(coefs) + 1

def get_features_as_c(features):
    # Create string from coefs for c
    features_string = "{"

    for i in range(len(features)):
        features_string += str(features[i])
        if i < len(features) - 1:
            features_string += ", "
    features_string += "}"

    return features_string

def save_c_file(code, model_type):
     # Save in c file
    file_name = model_type + ".c"

    print("Creating file: " + file_name)
    with open(file_name, "w+") as f:
        f.write(code)

    # Return command to execute
    cpml = "gcc -o {} {}".format(model_type, file_name)
    cmd = "./{}".format(model_type)

    return (cpml, cmd)

# Create C code
def produce_model_pred_c(model, features, model_type):
    
    c_code = """#include <stdio.h>\n"""
    coefs_string, coef_len = get_coefs_as_c(model, model_type)

    features_code = get_features_as_c(features)

    if model_type == "linear_regression":
        c_code += """/* Predicting linear regression from trained model. */

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
        float features_arr[2] = """ + features_code + """;
        float coef_arr[""" + str(coef_len)+ """] = """ + coefs_string + """;
        float *features = features_arr;
        float *coefs = coef_arr;
        int n_coefs = """ + str(coef_len) + """;

        /* Linear regression pred */
        printf("Prediction: %f", linear_regression_prediction(features, coefs, n_coefs));

        return 0;
    }"""

    elif model_type == "logistic_regression":
        c_code += """/* Predicting linear regression from trained model. */

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
        float features_arr[2] = """ + features_code + """;
        float coef_arr[""" + str(coef_len) + """] = """ + coefs_string + """;
        float *features = features_arr;
        float *coefs = coef_arr;
        int n_coefs = """ + str(coef_len) + """;

        /* Logistic regression pred */
        printf("Prediction: %f", logistic_regression(features, coefs, n_coefs));

        return 0;
    }"""

    elif model_type == "random_forest":
        pass

    return save_c_file(c_code, model_type)

# Compiling and running binary
def run(cpml, cmd):
    print("Compiling: " + cpml)
    subprocess.check_call(cpml, shell=True)

    print("Running: " + cmd)
    output = subprocess.check_output(cmd, shell=True)

    print("Output: " + str(output))

if __name__ == "__main__":
    # Get parameters as input
    features = [3.141592, 2.714836]
    model_type = input("Enter regression type (logistic or linear): ")

    while model_type not in ["linear", "logistic"]:
        model_type = input("Enter regression type (logistic or linear): ")

    model_type += "_regression"

    model_file = model_type + ".joblib"

    # Get model back
    model = joblib.load(model_file)

    # Launch transpile and compile c file
    run(*produce_model_pred_c(model, features, model_type))

    # Check if model as same value
    test_arr =np.asarray(features)
    pred = model.predict(test_arr.reshape(1, -1))

    if model_type == "logistic_regression":
        pred = sigmoid(pred)
        pred = 1 if pred > 0.5 else 0

    # Print model prediction
    print("Model predict: " + str(pred))
