#include "utility.h"
#include <cfenv>

ThreadPool thread_pool;

void ini_xinfo(matrix<double> &X, size_t N, size_t p)
{
    // matrix<double> X;
    X.resize(p);

    for (size_t i = 0; i < p; i++)
    {
        X[i].resize(N);
    }

    // return std::move(X);
    return;
}

void ini_xinfo(matrix<double> &X, size_t N, size_t p, double var)
{
    // matrix<double> X;
    X.resize(p);

    for (size_t i = 0; i < p; i++)
    {
        X[i].resize(N, var);
    }

    // return std::move(X);
    return;
}

void ini_xinfo_sizet(matrix<size_t> &X, size_t N, size_t p)
{
    // matrix<size_t> X;
    X.resize(p);

    for (size_t i = 0; i < p; i++)
    {
        X[i].resize(N);
    }

    // return std::move(X);
    return;
}

void row_sum(matrix<double> &X, std::vector<double> &output)
{
    size_t p = X.size();
    size_t N = X[0].size();
    // std::vector<double> output(N);
    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < p; j++)
        {
            // COUT << X[j][i] << endl;
            output[i] = output[i] + X[j][i];
        }
    }
    return;
}

void col_sum(matrix<double> &X, std::vector<double> &output)
{
    size_t p = X.size();
    size_t N = X[0].size();
    // std::vector<double> output(p);
    for (size_t i = 0; i < p; i++)
    {
        for (size_t j = 0; j < N; j++)
        {
            output[i] = output[i] + X[i][j];
        }
    }
    return;
}

double sum_squared(std::vector<double> &v)
{
    size_t N = v.size();
    double output = 0.0;
    for (size_t i = 0; i < N; i++)
    {
        output = output + pow(v[i], 2);
    }
    return output;
}

double sum_vec(std::vector<double> &v)
{
    size_t N = v.size();
    double output = 0;
    for (size_t i = 0; i < N; i++)
    {
        output = output + v[i];
    }
    return output;
}

void seq_gen_std(size_t start, size_t end, size_t length_out, std::vector<size_t> &vec)
{
    // generate a sequence of integers, save in std vector container
    double incr = (double)(end - start) / (double)length_out;

    for (size_t i = 0; i < length_out; i++)
    {
        vec[i] = (size_t)incr * i + start;
    }

    return;
}

void seq_gen_std2(size_t start, size_t end, size_t length_out, std::vector<size_t> &vec)
{
    // generate a sequence of integers, save in std vector container
    // different from seq_gen_std
    // always put the first element 0, actual vector output have length length_out + 1!
    double incr = (double)(end - start) / (double)length_out;

    vec[0] = 0;

    for (size_t i = 1; i < length_out + 1; i++)
    {
        vec[i] = (size_t)incr * (i - 1) + start;
    }

    return;
}

void vec_sum(std::vector<double> &vector, double &sum)
{
    sum = 0.0;
    for (size_t i = 0; i < vector.size(); i++)
    {
        sum = sum + vector[i];
    }
    return;
}

void vec_sum_sizet(std::vector<size_t> &vector, size_t &sum)
{
    sum = 0;
    for (size_t i = 0; i < vector.size(); i++)
    {
        sum = sum + vector[i];
    }
    return;
}

double sq_vec_diff(std::vector<double> &v1, std::vector<double> &v2)
{
    assert(v1.size() == v2.size());
    size_t N = v1.size();
    double output = 0.0;
    for (size_t i = 0; i < N; i++)
    {
        output = output + pow(v1[i] - v2[i], 2);
    }
    return output;
}

double sq_vec_diff_sizet(std::vector<size_t> &v1, std::vector<size_t> &v2)
{
    assert(v1.size() == v2.size());
    size_t N = v1.size();
    double output = 0.0;
    for (size_t i = 0; i < N; i++)
    {
        output = output + pow(v1[i] - v2[i], 2);
    }
    return output;
}

void unique_value_count2(const double *Xpointer, matrix<size_t> &Xorder_std, //std::vector<size_t> &X_values,
                         std::vector<double> &X_values, std::vector<size_t> &X_counts, std::vector<size_t> &variable_ind, size_t &total_points, std::vector<size_t> &X_num_unique, size_t &p_categorical, size_t &p_continuous)
{
    size_t N = Xorder_std[0].size();
    size_t p = Xorder_std.size();
    double current_value = 0.0;
    size_t count_unique = 0;
    size_t N_unique;
    variable_ind[0] = 0;

    total_points = 0;
    for (size_t i = p_continuous; i < p; i++)
    {
        // only loop over categorical variables
        // suppose p = (p_continuous, p_categorical)
        // index starts from p_continuous
        X_counts.push_back(1);
        current_value = *(Xpointer + i * N + Xorder_std[i][0]);
        X_values.push_back(current_value);
        count_unique = 1;

        for (size_t j = 1; j < N; j++)
        {
            if (*(Xpointer + i * N + Xorder_std[i][j]) == current_value)
            {
                X_counts[total_points]++;
            }
            else
            {
                current_value = *(Xpointer + i * N + Xorder_std[i][j]);
                X_values.push_back(current_value);
                X_counts.push_back(1);
                count_unique++;
                total_points++;
            }
        }
        variable_ind[i + 1 - p_continuous] = count_unique + variable_ind[i - p_continuous];
        X_num_unique[i - p_continuous] = count_unique;
        total_points++;
    }

    return;
}

double normal_density(double y, double mean, double var, bool take_log)
{
    // density of normal distribution
    double output = 0.0;

    output = -0.5 * log(2.0 * 3.14159265359 * var) - pow(y - mean, 2) / 2.0 / var;
    if (!take_log)
    {
        output = exp(output);
    }
    return output;
}

bool is_non_zero(size_t x) { return (x > 0); }

size_t count_non_zero(std::vector<double> &vec)
{
    size_t output = 0;
    for (size_t i = 0; i < vec.size(); i++)
    {
        if (vec[i] != 0)
        {
            output++;
        }
    }
    return output;
}

double update_delta(size_t tree_ind, matrix<double> &delta_loglike, std::vector<double> delta_cand, double dim_residual,  matrix<double> theta_vector, double concn, std::mt19937 &gen)
{
    std::feclearexcept(FE_OVERFLOW);
    std::feclearexcept(FE_UNDERFLOW);

    // Update delta_loglike for current trees
    size_t K = delta_cand.size(); // number of delta candiates
    size_t B = theta_vector.size();

    // break log-likelihood into parts
    double ret1 = 0;
    double ret2 = B * (dim_residual * concn * log(concn) - (dim_residual - 1) * lgamma(concn) - log(dim_residual));
    std::vector<double> ret3(K, 0.0);
    std::vector<double> temp(dim_residual, 0.0);
    // double temp_max = -INFINITY;
    // double temp_sum = 0.0;
    
    for(size_t b = 0; b < B; b++)
    {
        for(size_t j = 0; j < dim_residual; j++)
        {
            ret1 += (concn - 1) * log(theta_vector[b][j]) - concn * theta_vector[b][j];
        }

        for(size_t i = 0; i< K; i++)
        {
            // temp_max = pow(theta_vector[b][0], concn * (delta_cand[i] - 1));
            for(size_t j = 0; j < dim_residual; j++)
            {
                temp[j] = exp(concn * (delta_cand[i]-1) * log(theta_vector[b][j]) - lgamma(concn * delta_cand[i]));
            }
            vec_sum(temp, temp_sum);
            ret3[i] += log(temp_sum); //+ log(temp_max); 
            // improve this to avoid the sum going to inf         
        }
    }
    
    for(size_t i = 0; i < K; i++)
    {
        delta_loglike[tree_ind][i] = ret1 + ret2 + ret3[i] + B * concn * (delta_cand[i] - 1) * log(concn);
        // cout << "delta = " << delta_cand[i] << " likelihood " << delta_loglike[tree_ind][i] << endl;

        if((bool)std::fetestexcept(FE_OVERFLOW)) 
        {
            cout << "likelihood overflows for delta " << delta_cand[i] << endl;
            abort();
        }
        else if((bool)std::fetestexcept(FE_UNDERFLOW))
        {
            cout << "likelihood underflows for delta " << delta_cand[i] << endl;
            abort();
        }
    }

    // Draw Delta
    
    std::vector<double> delta_likelihood(K, 0.0);
    double loglike_max = -INFINITY;

    for (size_t i = 0; i < K; i++)
    {
        temp_delta_loglike[i] += delta_loglike[tree_ind][i];
        if (temp_delta_loglike[i] > loglike_max){ loglike_max = temp_delta_loglike[i]; }
    }
    for (size_t i = 0; i < K; i++)
    {
        delta_likelihood[i] = exp(temp_delta_loglike[i] - loglike_max);
    }
    std::discrete_distribution<> d(delta_likelihood.begin(), delta_likelihood.end());
    return(delta_cand[d(gen)]);
    // std::cout << "delta likeihood " << delta_likelihood << endl;
    // std::cout << "delta " << state->sigma << endl;

    // std::fill(delta_likelihood.begin(), delta_likelihood.end(), 1.0);
    // std::discrete_distribution<> g(delta_likelihood.begin(), delta_likelihood.end());
    // state->update_sigma(delta_cand[g(state->gen)]);
    // std::cout << "fake delta " << state->sigma << endl;

}
