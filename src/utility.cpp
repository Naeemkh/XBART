#include "utility.h"
#include "gsl/gsl_integration.h"
#include "gsl/gsl_errno.h"
#include <boost/math/tools/roots.hpp>
#include <boost/math/distributions/lognormal.hpp>
#include <boost/bind.hpp>

// ThreadPool thread_pool;
// 
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

    std::cout << "total_points " << total_points << std::endl;

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

    double LogitKernel(double x, void * params)
	{   
        LogitParams p = * (LogitParams *) params;
        return exp( (p.w * p.r + p.tau_a) * log(x) - p.s * pow(x, p.w) - p.tau_b * x - p.logv);
        // return exp( (p.w * p.r + p.tau_a) * log(x / p.mx) - p.s *  (pow(x, p.w) - pow(p.mx, p.w)) - p.tau_b * (x - p.mx) );
    }

    double log_logit_kernel(double x, void *params)
    {
        LogitParams p = * (LogitParams *) params;

        return (p.w * p.r + p.tau_a) * log(x) - p.s * pow(x, p.w) - p.tau_b * x;
    }

    double derive_logit_kernel(double x, void *params)
    {
        LogitParams p = * (LogitParams *) params;

        return p.tau_b * x + p.s * p.w * pow(x, p.w) + 1 - p.tau_a - p.r * p.w;
    }


    struct TerminationCondition  {
        bool operator() (double min, double max)  {
        return abs(min - max) <= 0.000001;
        }
    };

    int get_root(double (*kernel)(double x, void *params), void *params, double &mx, double const &by, double const &limit, double const &tol)
    {
        std::pair<double, double> range(0.0, by);
        std::pair<double, double> mx_bisect;
        // boost::math::tools::eps_tolerance<double> eps_tol(tol);
        while (range.second <= limit)
        {
            try
            {
                mx_bisect = boost::math::tools::bisect(boost::bind(kernel, _1, params), range.first, range.second, TerminationCondition());
                // cout << "mx_bisect (" << mx_bisect.first << ", " << mx_bisect.second << endl;
                mx = (mx_bisect.first + mx_bisect.second) / 2;
                if (mx == 0)
                {cout << "mx = 0, mx_bisect (" << mx_bisect.first << ", " << mx_bisect.second << ")" << endl;}
                return 0;
            }
            catch(const std::exception& e)
            {
                    range.first += by;
                    range.second += by;
            }   
        }
        // std::cerr << e.what() << '\n';
        // cout << "can't find root up to " << limit << endl;
        return 1;

    }

    int get_integration(double (*kernel)(double x, void *params), void *params, double &output, double const &lower_bound)
    {
        
        gsl_function F;
        F.function = kernel;
        F.params = params;
        double error;

        gsl_set_error_handler_off();
        gsl_integration_workspace *workspace = gsl_integration_workspace_alloc(3000);
        // gsl_integration_qagiu(function, lower_limit, absolut error limit, relatvie error limit, max # of sub interval, workspace, result, error)
        int status = gsl_integration_qagiu(&F, lower_bound, 0, 1e-6, 2000, workspace, &output, &error);
        gsl_integration_workspace_free(workspace);

        // if(take_log)
        // {
        //     output = log(output) + logv;
        // }
        // else
        // {
        //     output = exp(logv) * output;
        // }
        
        return status;
            
    }