#include <Eigen/Dense>
#include <Eigen/src/Core/Product.h>

#include <iostream>
#include <fstream>
#include <vector>

/*
TODO
    Improve learning rate decay
    Normal Distribution for weight initialization
    Add Data Augmentation
    Figure out why Loss can get stuck at 0.2 > Seems to be when guesses all go to 0.1
    Find out why this is slower than Python
    Add Convolution
    Batch Normalization
    Adam
    Measure times of different parts of the program
    Get some of these chunks in to functions, such as Forward Pass
    Make np_dot and relu_grad handle bad sized matrices without hard fail
    Beat 95%

DONE
    Load MNIST Data
    Initialize Weights
    Forward Pass
    Backward Pass
    Calculate Loss
    Calculate Accuracy
    Xavier Initialization
    Get convergence on small slice of dataset
    Beat 90% on validation data
    Load data with functions
    Compare speed of float vs double (With Eigen the doubles seem slower than floats)
    Beat 92% on validation data (94.5%+)
    Xavier Initialization Function
*/

int reverseInt(int i) {
    // https://stackoverflow.com/questions/8286668/how-to-read-mnist-data-in-c
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

Eigen::MatrixXf np_dot(Eigen::MatrixXf error, Eigen::MatrixXf weights) {
    // Basically a Numpy Dot Product. Eigen doesn't seem to do this but who knows maybe they do.
    Eigen::MatrixXf output(1, weights.cols());
    for (int i = 0; i < weights.cols(); i++) {
        float thesum = 0;
        for (int j = 0; j < weights.rows(); j++) {
            thesum += error(j) * weights(j, i);
        }
        output(i) = thesum;
    }
    return output;
}

Eigen::MatrixXf relu_grad(Eigen::MatrixXf error, Eigen::MatrixXf res_rel) {
    // error = np.dot(out.T, error) * (res_rel1 > 0)     This is the (res_rel > 0) portion
    Eigen::MatrixXf output(1, res_rel.rows());
    for (int i = 0; i < res_rel.rows(); i++) {
        if (res_rel(i) > 0) {
            output(i) = error(i);
        }
        else {
            output(i) = 0;
        }
    }
    return output;
}

std::vector<Eigen::MatrixXi> load_y_data(const char *gotstring, int runs) {
    /*
    ######################################################
                       LOAD Y DATA
    ######################################################
    */
    std::vector<Eigen::MatrixXi> Y;
    std::ifstream filey(gotstring, std::ios::binary);
    if (filey.is_open())
    {
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        filey.read((char*)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);
        filey.read((char*)&number_of_images, sizeof(number_of_images));
        number_of_images = reverseInt(number_of_images);
        std::cout << number_of_images << std::endl;
        int label_int;
        for (int i = 0; i < runs; i++)
        {
            Eigen::MatrixXi label;
            label = Eigen::MatrixXi::Zero(1, 10);
            unsigned char temp = 0;
            filey.read((char*)&temp, sizeof(temp));
            label_int = temp;
            label(0, label_int) = 1;
            Y.push_back(label);
        }
        filey.close();
        return Y;
    }
    else {
        std::cout << "FAILED TO LOAD THE FILE " << gotstring << std::endl;;
        Y.push_back(Eigen::MatrixXi::Zero(1, 10));
        return Y; // Send zeros if load fails
    }
}

std::vector<Eigen::MatrixXf> load_x_data(const char* gotstring, int runs) {
    /*
    ######################################################
                       LOAD X DATA
    ######################################################
    */
    std::vector<Eigen::MatrixXf> X;
    // LOAD THE MNIST DATA http://yann.lecun.com/exdb/mnist/
    // MNIST Data https://stackoverflow.com/questions/23253485/little-endian-reading-mnist-file-numbers-out-of-range
    // MNIST Data https://stackoverflow.com/questions/8286668/how-to-read-mnist-data-in-c
    // MNIST Data https://stackoverflow.com/questions/12993941/how-can-i-read-the-mnist-dataset-with-c
    // MNIST Data https://stackoverflow.com/questions/16871512/how-to-read-pixels-from-mnist-digit-database-and-create-the-iplimage?noredirect=1&lq=1
    // 3D Eigen https://stackoverflow.com/questions/17098218/most-efficient-option-for-build-3d-structures-using-eigen-matrices
    std::ifstream filex(gotstring, std::ios::binary);
    if (filex.is_open())
    {
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        filex.read((char*)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);
        filex.read((char*)&number_of_images, sizeof(number_of_images));
        number_of_images = reverseInt(number_of_images);
        std::cout << number_of_images << std::endl;
        filex.read((char*)&n_rows, sizeof(n_rows));
        n_rows = reverseInt(n_rows);
        std::cout << "n_rows = " << n_rows << std::endl;
        filex.read((char*)&n_cols, sizeof(n_cols));
        n_cols = reverseInt(n_cols);
        std::cout << "n_cols = " << n_cols << std::endl;
        for (int i = 0; i < runs; i++)
        {
            Eigen::MatrixXf thisImage(28, 28);
            Eigen::MatrixXf thisImageFlat(1, 784);
            for (int r = 0; r < n_rows; ++r)
            {
                for (int c = 0; c < n_cols; ++c)
                {
                    unsigned char temp = 0;
                    filex.read((char*)&temp, sizeof(temp));
                    int mnist_int = temp;
                    float mnist_float = mnist_int / 255.0;
                    thisImage(r, c) = mnist_float;
                }
            }
            thisImage.resize(1, 784); // Flatten the image
            X.push_back(thisImage);
        }
        filex.close();
        return X;
    }
    else {
        std::cout << "FAILED TO LOAD THE FILE " << gotstring << std::endl;
        X.push_back(Eigen::MatrixXf::Zero(1, 784));
        return X; // Send zeros if load fails
    }
}

Eigen::MatrixXf box_muller(Eigen::MatrixXf mat) {
    /*
    From Uniform Distribution to a sort of triangle distribution
    https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
    */
    Eigen::MatrixXf second(mat.rows(), mat.cols()); // Box-Muller requires two sets of random data
    second = Eigen::MatrixXf::Random(mat.rows(), mat.cols());
    second = ((second.array() + 1) / (float)2);
    mat = sqrt(-2 * log(mat.array())) * cos(2 * 3.14 * second.array()); // Box-Muller
    for (int r = 0; r < mat.rows(); r++) { // Sometimes I get a NAN probably getting negative square roots due to Eigen's Random algo
        for (int c = 0; c < mat.cols(); c++) {
            if (mat(r, c) == mat(r, c)) {
            }
            else {
                mat(r, c) = 0.500;
            }
        }
    }
    return mat;
}

Eigen::MatrixXf xavier_initialization(Eigen::MatrixXf mat) {
    /*
    Xavier Initialization to keep weights sane during gradient descent
    */
    mat = mat.array() * sqrt(1 / (float)(mat.rows() + mat.cols()));
    return mat;
}

int main()
{
    // Init Training
    int div_by = 1; // Use this to run on smaller number of MNIST images
    int epochs = 2;
    float lr = 0.25;
    int runs = int(60000 / div_by);

    // Load Y Data
    std::vector<Eigen::MatrixXi> Y_train;
    Y_train = load_y_data("train-labels.idx1-ubyte", runs);
    if (Y_train[0].sum() == 0) {
        return 0; // Fail Elegantly
    }

    // Load X Data
    std::vector<Eigen::MatrixXf> X_train;
    X_train = load_x_data("train-images.idx3-ubyte", runs);
    if (X_train[0].sum() == 0) {
        return 0; // Fail Elegantly
    }
    
    // Layers
    Eigen::MatrixXf w0;
    Eigen::MatrixXf w1;
    Eigen::MatrixXf out;
    w0 = Eigen::MatrixXf::Random(64, 784); // Eigen only does Uniform -1 to 1
    w1 = Eigen::MatrixXf::Random(32, 64);
    out = Eigen::MatrixXf::Random(10, 32);

    w0 = ((w0.array() + 1) / (float)2); // Change to Uniform 0 to 1
    w1 = ((w1.array() + 1) / (float)2);
    out = ((out.array() + 1) / (float)2);

    w0 = box_muller(w0).array(); // Box-Muller
    w1 = box_muller(w1).array();
    out = box_muller(out).array();

    w0 = xavier_initialization(w0); //  Xavier Initialization
    w1 = xavier_initialization(w1);
    out = xavier_initialization(out);

    // Dx Initialization
    Eigen::MatrixXf old_dx_out;
    Eigen::MatrixXf old_dx_w1;
    Eigen::MatrixXf old_dx_w0;
    old_dx_out = Eigen::MatrixXf::Zero(10, 32);
    old_dx_w1 = Eigen::MatrixXf::Zero(32, 64);
    old_dx_w0 = Eigen::MatrixXf::Zero(64, 784);
    Eigen::MatrixXf vold_dx_out;
    Eigen::MatrixXf vold_dx_w1;
    Eigen::MatrixXf vold_dx_w0;
    vold_dx_out = Eigen::MatrixXf::Zero(10, 32);
    vold_dx_w1 = Eigen::MatrixXf::Zero(32, 64);
    vold_dx_w0 = Eigen::MatrixXf::Zero(64, 784);

    // Result Matrix Initialization
    Eigen::MatrixXf res_w0(64, 1);
    Eigen::MatrixXf res_rel0(64, 1);
    Eigen::MatrixXf res_w1(32, 1);
    Eigen::MatrixXf res_rel1(32, 1);
    Eigen::MatrixXf res_out(10, 1);
    Eigen::MatrixXf guess(10, 1);
    Eigen::MatrixXf error(10, 1);
    Eigen::MatrixXf softMaxValue(10, 1);

    /*
    ######################################################
                       Main Loop
    ######################################################
    */
    std::vector<int> corrects;
    std::vector<float> losses;
    for (int epoch = 0; epoch < epochs; epoch++) {
        float correct_percent;
        float loss;
        float old_loss = 0.0;
        int corrects_sum = 0;
        float lr_mod = 1.0;
        float this_lr = 0.0;
        for (int i = 0; i < runs; i++) {
            // Data Matrices
            Eigen::MatrixXf x(1, 784);
            x = X_train[i];
            Eigen::MatrixXi y(1, 10);
            y = Y_train[i];

            /*
            ######################################################
                              Forward Pass
            ######################################################
            */
            res_w0 << w0 * x.transpose();
            res_rel0 << res_w0.cwiseMax(0); // ReLU
            res_w1 << w1 * res_rel0;
            res_rel1 << res_w1.cwiseMax(0);
            res_out << out * res_rel1;
            softMaxValue = res_out.array() - res_out.maxCoeff();
            guess << exp(softMaxValue.array()) / (exp(softMaxValue.array())).sum(); // guess = np.exp(res_out - res_out.max()) / np.sum(np.exp(res_out - res_out.max()), axis=0
            
            error << guess(0) - y(0), guess(1) - y(1), guess(2) - y(2), guess(3) - y(3), guess(4) - y(4), guess(5) - y(5), guess(6) - y(6), guess(7) - y(7), guess(8) - y(8), guess(9) - y(9);

            // Loss and Accuracy
            loss = 0;
            for (int k = 0; k < 10; k++) {
                loss += abs(error(k));
            }
            loss = loss / (float)10.0;
            float guess_max = 0;
            int guess_idx = 0;
            for (int k = 0; k < 10; k++) {
                if (guess(k) > guess_max) {
                    guess_max = guess(k);
                    guess_idx = k;
                }
            }
            if (y(guess_idx) == 1) {
                corrects.push_back(1);
                corrects_sum += 1;
            }
            else {
                corrects.push_back(0);
            }

            /*
            ######################################################
                               Backward Pass
            ######################################################
            */
            Eigen::MatrixXf dd(10, 1);
            Eigen::MatrixXf dx_out(10, 32);
            Eigen::MatrixXf error_new(32,1);
            Eigen::MatrixXf dx_w1(32, 64);
            Eigen::MatrixXf error_final(64, 1);
            Eigen::MatrixXf dx_w0(64, 784);

            dd << guess.array() * (1 - guess.array()); // (10, 1)
            error << error(0) * dd(0), error(1) * dd(1), error(2) * dd(2), error(3) * dd(3), error(4) * dd(4), error(5) * dd(5), error(6) * dd(6), error(7) * dd(7), error(8) * dd(8), error(9) * dd(9); // (10, 1)    This isn't what I had in Python
            dx_out << error * res_rel1.transpose(); // (10, 32)
            error_new = relu_grad( np_dot(error.transpose(), out), res_rel1); // (1, 32)
            dx_w1 = error_new.transpose() * res_rel0.transpose(); // (32, 64)
            error_final = relu_grad(np_dot(error_new, w1), res_rel0); // (64, 1)
            dx_w0 = error_final.transpose() * x; // (64, 784)

            // Adjust Weights
            if (guess_max < 0.2) { // Sometimes it will guess all 0.1 so maybe this will nudge it away from doing that
                lr_mod = 1.5;
            }
            this_lr = lr * lr_mod;
            out = out.array() - this_lr * dx_out.array() - 0.25 * this_lr * old_dx_out.array() - 0.1 * this_lr * vold_dx_out.array();
            w0 = w0.array() - this_lr * dx_w0.array() - 0.25 * this_lr * old_dx_w0.array() - 0.1 * this_lr * vold_dx_w0.array();
            w1 = w1.array() - this_lr * dx_w1.array() - 0.25 * this_lr * old_dx_w1.array() - 0.1 * this_lr * vold_dx_w1.array();
            lr_mod = 1.0;

            // Adjust Learning Rate Over Time
            if (i > 0) {
                correct_percent = corrects_sum / (float)i;
                if (i % 1000 == 0) {
                    std::cout << std::endl << "Correct Percent = " << correct_percent << " loss = " << loss << " Epoch = " << epoch << " img_index = " << i;
                }
                if (i > 1000 && correct_percent > 0.4 && correct_percent <= 0.50) {
                    lr = 0.15;
                }
                else if (i > 2000 && correct_percent > 0.50 && correct_percent <= 0.60) {
                    lr = 0.15;
                }
                else if (i > 2000 && correct_percent > 0.60 && correct_percent <= 0.74) {
                    lr = 0.075;
                }
                else if (i > 2000 && correct_percent > 0.74 && correct_percent <= 0.87) {
                    lr = 0.05;
                }
                else if (i > 2000 && correct_percent > 0.87 && correct_percent <= 0.95) {
                    lr = 0.0025;
                }
            }

            // Momentum Updates
            vold_dx_out = old_dx_out;
            vold_dx_w1 = old_dx_w1;
            vold_dx_w0 = old_dx_w0;
            old_dx_out = dx_out;
            old_dx_w1 = dx_w1;
            old_dx_w0 = dx_w0;
            old_loss = loss;
        }
        std::cout << std::endl << "==================================";
        std::cout << std::endl << "Correct Percent = " << correct_percent;
        std::cout << std::endl << "==================================";
    }

    /*
    =====================================================================================================================================================
    End of Training
    ------------------------------
    Begin Validation
    ===================================================================================================================================================== 
    */

    // Init Validation
    div_by = 1;
    runs = int(10000 / div_by);

    // Load Y Data
    std::vector<Eigen::MatrixXi> Y_test;
    Y_test = load_y_data("t10k-labels.idx1-ubyte", runs);
    if (Y_test[0].sum() == 0) {
        return 0; // Fail Elegantly
    }

    // Load X Data
    std::vector<Eigen::MatrixXf> X_test;
    X_test = load_x_data("t10k-images.idx3-ubyte", runs);
    if (X_test[0].sum() == 0) {
        return 0; // Fail Elegantly
    }

    float loss = 0.0;
    int corrects_sum = 0;
    for (int i = 0; i < runs; i++) {
        // Data Matrices
        Eigen::MatrixXf x(1, 784);
        x = X_test[i];
        Eigen::MatrixXi y(1, 10);
        y = Y_test[i];

        /*
        ######################################################
                          Forward Pass
        ######################################################
        */
        res_w0 << w0 * x.transpose();
        res_rel0 << res_w0.cwiseMax(0); // ReLU
        res_w1 << w1 * res_rel0;
        res_rel1 << res_w1.cwiseMax(0);
        res_out << out * res_rel1;
        softMaxValue = res_out.array() - res_out.maxCoeff();
        guess << exp(softMaxValue.array()) / (exp(softMaxValue.array())).sum(); // guess = np.exp(res_out - res_out.max()) / np.sum(np.exp(res_out - res_out.max()), axis=0)
        error << guess(0) - y(0), guess(1) - y(1), guess(2) - y(2), guess(3) - y(3), guess(4) - y(4), guess(5) - y(5), guess(6) - y(6), guess(7) - y(7), guess(8) - y(8), guess(9) - y(9);

        // Loss and Accuracy
        loss = 0;
        for (int k = 0; k < 10; k++) {
            loss += abs(error(k));
        }
        loss = loss / (float)10.0;
        float guess_max = 0;
        int guess_idx = 0;
        for (int k = 0; k < 10; k++) {
            if (guess(k) > guess_max) {
                guess_max = guess(k);
                guess_idx = k;
            }
        }
        if (y(guess_idx) == 1) {
            corrects.push_back(1);
            corrects_sum += 1;
        }
        else {
            corrects.push_back(0);
        }
    }
    float correct_percent = corrects_sum / (float)runs;
    std::cout << std::endl;
    std::cout << std::endl << "####################################################################";
    std::cout << std::endl << "               " << correct_percent;
    std::cout << std::endl << "####################################################################";
    std::cout << std::endl;
    return 0;
}
