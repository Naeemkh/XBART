#include "tree.h"
#include "model.h"
#include <cfenv>
#include <functional>
#include <boost/math/tools/roots.hpp>
#include <boost/math/distributions/lognormal.hpp>
#include <boost/bind.hpp>

//////////////////////////////////////////////////////////////////////////////////////
//
//
//  Normal Model
//
//
//////////////////////////////////////////////////////////////////////////////////////

void NormalModel::incSuffStat(matrix<double> &residual_std, size_t index_next_obs, std::vector<double> &suffstats)
{
    // I have to pass matrix<double> &residual_std, size_t index_next_obs
    // which allows more flexibility for multidimensional residual_std

    suffstats[0] += residual_std[0][index_next_obs];
    return;
}

void NormalModel::samplePars(std::unique_ptr<State> &state, std::vector<double> &suff_stat, std::vector<double> &theta_vector, double &prob_leaf, const size_t &tree_ind)
{
    std::normal_distribution<double> normal_samp(0.0, 1.0);

    // test result should be theta
    theta_vector[0] = suff_stat[0] / pow(state->sigma, 2) / (1.0 / tau + suff_stat[2] / pow(state->sigma, 2)) + sqrt(1.0 / (1.0 / tau + suff_stat[2] / pow(state->sigma, 2))) * normal_samp(state->gen); //Rcpp::rnorm(1, 0, 1)[0];//* as_scalar(arma::randn(1,1));

    // also update probability of leaf parameters
    // prob_leaf = normal_density(theta_vector[0], suff_stat[0] / pow(state->sigma, 2) / (1.0 / tau + suff_stat[2] / pow(state->sigma, 2)), 1.0 / (1.0 / tau + suff_stat[2] / pow(state->sigma, 2)), true);

    return;
}

void NormalModel::update_state(std::unique_ptr<State> &state, size_t tree_ind, std::unique_ptr<X_struct> &x_struct)
{
    // Draw Sigma
    // state->residual_std_full = state->residual_std - state->predictions_std[tree_ind];

    // residual_std is only 1 dimensional for regression model

    std::vector<double> full_residual(state->n_y);

    for (size_t i = 0; i < state->residual_std[0].size(); i++)
    {
        full_residual[i] = state->residual_std[0][i] - (*(x_struct->data_pointers[tree_ind][i]))[0];
    }

    std::gamma_distribution<double> gamma_samp((state->n_y + kap) / 2.0, 2.0 / (sum_squared(full_residual) + s));
    state->update_sigma(1.0 / sqrt(gamma_samp(state->gen)));
    return;
}

void NormalModel::initialize_root_suffstat(std::unique_ptr<State> &state, std::vector<double> &suff_stat)
{
    // sum of y
    suff_stat[0] = sum_vec(state->residual_std[0]);
    // sum of y squared
    suff_stat[1] = sum_squared(state->residual_std[0]);
    // number of observations in the node
    suff_stat[2] = state->n_y;
    return;
}

void NormalModel::updateNodeSuffStat(std::vector<double> &suff_stat, matrix<double> &residual_std, matrix<size_t> &Xorder_std, size_t &split_var, size_t row_ind)
{
    suff_stat[0] += residual_std[0][Xorder_std[split_var][row_ind]];
    suff_stat[1] += pow(residual_std[0][Xorder_std[split_var][row_ind]], 2);
    suff_stat[2] += 1;
    return;
}

void NormalModel::calculateOtherSideSuffStat(std::vector<double> &parent_suff_stat, std::vector<double> &lchild_suff_stat, std::vector<double> &rchild_suff_stat, size_t &N_parent, size_t &N_left, size_t &N_right, bool &compute_left_side)
{

    // in function split_xorder_std_categorical, for efficiency, the function only calculates suff stat of ONE child
    // this function calculate the other side based on parent and the other child

    if (compute_left_side)
    {
        rchild_suff_stat = parent_suff_stat - lchild_suff_stat;
    }
    else
    {
        lchild_suff_stat = parent_suff_stat - rchild_suff_stat;
    }
    return;
}

void NormalModel::state_sweep(size_t tree_ind, size_t M, matrix<double> &residual_std, std::unique_ptr<X_struct> &x_struct) const
{
    size_t next_index = tree_ind + 1;
    if (next_index == M)
    {
        next_index = 0;
    }

    ////////////////////////////////////////////////////////
    // Be care of line 151 in train_all.cpp, initial_theta
    ////////////////////////////////////////////////////////

    for (size_t i = 0; i < residual_std[0].size(); i++)
    {
        residual_std[0][i] = residual_std[0][i] - (*(x_struct->data_pointers[tree_ind][i]))[0] + (*(x_struct->data_pointers[next_index][i]))[0];
    }
    return;
}

double NormalModel::likelihood(std::vector<double> &temp_suff_stat, std::vector<double> &suff_stat_all, size_t N_left, bool left_side, bool no_split, std::unique_ptr<State> &state) const
{
    // likelihood equation,
    // note the difference of left_side == true / false
    // node_suff_stat is mean of y, sum of square of y, saved in tree class
    // double y_sum = (double)suff_stat_all[2] * suff_stat_all[0];
    // double y_sum = suff_stat_all[0];
    double sigma2 = state->sigma2;
    double ntau;
    // double suff_one_side;

    /////////////////////////////////////////////////////////////////////////
    //
    //  I know combining likelihood and likelihood_no_split looks nicer
    //  but this is a very fundamental function, executed many times
    //  the extra if(no_split) statement and value assignment make the code about 5% slower!!
    //
    /////////////////////////////////////////////////////////////////////////

    size_t nb;
    double nbtau;
    double y_sum;
    double y_squared_sum;

    if (no_split)
    {
        // ntau = suff_stat_all[2] * tau;
        // suff_one_side = y_sum;

        nb = suff_stat_all[2];
        nbtau = nb * tau;
        y_sum = suff_stat_all[0];
        y_squared_sum = suff_stat_all[1];
    }
    else
    {
        if (left_side)
        {
            nb = N_left + 1;
            nbtau = nb * tau;
            // ntau = (N_left + 1) * tau;
            y_sum = temp_suff_stat[0];
            y_squared_sum = temp_suff_stat[1];
            // suff_one_side = temp_suff_stat[0];
        }
        else
        {
            nb = suff_stat_all[2] - N_left - 1;
            nbtau = nb * tau;
            y_sum = suff_stat_all[0] - temp_suff_stat[0];
            y_squared_sum = suff_stat_all[1] - temp_suff_stat[1];

            // ntau = (suff_stat_all[2] - N_left - 1) * tau;
            // suff_one_side = y_sum - temp_suff_stat[0];
        }
    }

    // return 0.5 * log(sigma2) - 0.5 * log(nbtau + sigma2) + 0.5 * tau * pow(y_sum, 2) / (sigma2 * (nbtau + sigma2));

    return -0.5 * nb * log(2 * 3.141592653) - 0.5 * nb * log(sigma2) + 0.5 * log(sigma2) - 0.5 * log(nbtau + sigma2) - 0.5 * y_squared_sum / sigma2 + 0.5 * tau * pow(y_sum, 2) / (sigma2 * (nbtau + sigma2));
}

// double NormalModel::likelihood_no_split(std::vector<double> &suff_stat, std::unique_ptr<State> &state) const
// {
//     // the likelihood of no-split option is a bit different from others
//     // because the sufficient statistics is y_sum here
//     // write a separate function, more flexibility
//     double ntau = suff_stat[2] * tau;
//     // double sigma2 = pow(state->sigma, 2);
//     double sigma2 = state->sigma2;
//     double value = suff_stat[2] * suff_stat[0]; // sum of y

//     return 0.5 * log(sigma2) - 0.5 * log(ntau + sigma2) + 0.5 * tau * pow(value, 2) / (sigma2 * (ntau + sigma2));
// }

void NormalModel::ini_residual_std(std::unique_ptr<State> &state)
{
    // initialize partial residual at (num_tree - 1) / num_tree * yhat
    double value = state->ini_var_yhat * ((double)state->num_trees - 1.0) / (double)state->num_trees;
    for (size_t i = 0; i < state->residual_std[0].size(); i++)
    {
        state->residual_std[0][i] = (*state->y_std)[i] - value;
    }
    return;
}

void NormalModel::predict_std(const double *Xtestpointer, size_t N_test, size_t p, size_t num_trees, size_t num_sweeps, matrix<double> &yhats_test_xinfo, vector<vector<tree>> &trees)
{

    matrix<double> output;

    // row : dimension of theta, column : number of trees
    ini_matrix(output, this->dim_theta, trees[0].size());

    for (size_t sweeps = 0; sweeps < num_sweeps; sweeps++)
    {
        for (size_t data_ind = 0; data_ind < N_test; data_ind++)
        {
            getThetaForObs_Outsample(output, trees[sweeps], data_ind, Xtestpointer, N_test, p);

            // take sum of predictions of each tree, as final prediction
            for (size_t i = 0; i < trees[0].size(); i++)
            {
                yhats_test_xinfo[sweeps][data_ind] += output[i][0];
            }
        }
    }
    return;
}

//////////////////////////////////////////////////////////////////////////////////////
//
//
//  Logit Model
//
//
//
//////////////////////////////////////////////////////////////////////////////////////

//incSuffStat should take a state as its first argument
void LogitModel::incSuffStat(matrix<double> &residual_std, size_t index_next_obs, std::vector<double> &suffstats)
{
    // I have to pass matrix<double> &residual_std, size_t index_next_obs
    // which allows more flexibility for multidimensional residual_std

    // suffstats[0] += residual_std[0][index_next_obs];

    // sufficient statistics have 2 * num_classes

    suffstats[(*y_size_t)[index_next_obs]] += 1; //weight;


    for (size_t j = 0; j < dim_theta; ++j)
    {
        // count number of observations, y_{ij}
        // if ((*y_size_t)[index_next_obs] == j)
            // suffstats[j] += 1;

        // psi * f
        suffstats[dim_theta + j] += (*phi)[index_next_obs] *  pow(residual_std[j][index_next_obs], weight);
        // if (isnan(suffstats[dim_theta + j])) {cout << "phi = " << (*phi)[index_next_obs] << "; resid = " << residual_std[j][index_next_obs] << "; j = " << j << endl; }
    }

    return;
}

void LogitModel::samplePars(std::unique_ptr<State> &state, std::vector<double> &suff_stat, std::vector<double> &theta_vector, double &prob_leaf, const size_t &tree_ind)
{
    size_t c = dim_theta;
    LogitParams *lparams = new LogitParams(tau_a, tau_b, weight, 0.0, 0.0);
    double mx, output, logk, mval, M, theta , u, sigma;
    size_t count_reject;
    size_t reject_limit = 100;

    for (size_t j = 0; j < dim_theta; j++)
    {
        if (weight == 1 | suff_stat[c + j] > tau_b + 500)
        {
            std::gamma_distribution<double> gammadist(tau_a + suff_stat[j] +1/weight, 1.0);

            theta_vector[j] =  pow(gammadist(state->gen) / (tau_b + suff_stat[c + j]), 1 / weight ) ;
        }
        else if (suff_stat[c + j] == 0)
        {
            std::gamma_distribution<double> gammadist(tau_a + suff_stat[j] * weight);
            theta_vector[j] = gammadist(state->gen) / (tau_b);
        }
        else
        {
            lparams->set_r(suff_stat[j]);
            lparams->set_s(suff_stat[c + j]);
            lparams->set_logv(1);
            
            int status_mx = get_root(derive_logit_kernel, lparams, mx,  5.0, 10000, 1e-6); // status_mx = 1 if can't find root
            if ( !status_mx) // set logv if find root
            { 
                // lparams->set_mx(mx);
                lparams->set_logv(log_logit_kernel(mx, lparams));
            } 
            else
            {
                lparams->print();
                cout << "can't find root in samplePars, use general Gamma"  << endl;   
                // need to reconsider, how to sample when we can't find root??
                std::gamma_distribution<double> gammadist(tau_a + suff_stat[j] +1/weight, 1.0);
                theta_vector[j] =  pow(gammadist(state->gen) / (tau_b + suff_stat[dim_theta + j]), 1 / weight ) ;
                continue;
            }
            
            int status = get_integration(LogitKernel, lparams, output, 0.5*mx);

            if (output == 0)
            {
                lparams->print();
                cout << "integration = 0 in samplePars, use general Gamma" << endl;
                std::gamma_distribution<double> gammadist(tau_a + suff_stat[j] +1/weight, 1.0);
                theta_vector[j] =  pow(gammadist(state->gen) / (tau_b + suff_stat[dim_theta + j]), 1 / weight ) ;
                continue;
            }
            else if (status)
            {
                lparams->print();
                cout << "integration failed in samplePars. output = " << output << ", use general Gamma" << endl;
                std::gamma_distribution<double> gammadist(tau_a + suff_stat[j] +1/weight, 1.0);
                theta_vector[j] =  pow(gammadist(state->gen) / (tau_b + suff_stat[dim_theta + j]), 1 / weight ) ; 
                continue;
            }
            else
            {
                logk = log(output) + lparams->logv;
                mval = LogitKernel(mx, lparams) / output;
                // sigma = 1 / sqrt(weight * suff_stat[c+j] + tau_b);
                sigma = 1 / weight / sqrt(suff_stat[c + j] + tau_b);

                boost::math::lognormal_distribution<double> dlnorm(log(mx), sigma);
                std::lognormal_distribution<double> rlnorm(log(mx), sigma);
                std::uniform_real_distribution<double> runif(0.0,1.0);

                M = mval / pdf(dlnorm, exp(log(mx))); 

                count_reject = 0;
                while(count_reject < reject_limit)
                {
                    theta = rlnorm(state->gen);
                    u = runif(state->gen);
                    if (u < exp(log_logit_kernel(theta, lparams) - lparams->logv - logk - log(pdf(dlnorm, theta)) - log(M)))
                    {
                        theta_vector[j] = theta;
                        break;
                    }
                    count_reject += 1;
                }
                if (count_reject >= reject_limit)
                {
                    lparams->print();
                    cout << "warning: reject sampling after " <<  reject_limit << " iterations, use general Gamma" << endl;  
                    std::gamma_distribution<double> gammadist(tau_a + suff_stat[j] +1/weight, 1.0);
                    theta_vector[j] =  pow(gammadist(state->gen) / (tau_b + suff_stat[c + j]), 1 / weight ) ;
                }
                
            }     
        }
    }

    return;
}

void LogitModel::update_state(std::unique_ptr<State> &state, size_t tree_ind, std::unique_ptr<X_struct> &x_struct)
{
    double max_loglike_weight = -INFINITY;
    std::vector<double> loglike_weight(weight_std.size(), 0.0);
    std::vector<double> fits_w(dim_residual, 0.0);

    // rewrite loglike_weight calculation
    for (size_t j = 0; j < weight_std.size(); j++)
    {
        #pragma omp task shared(loglike_weight, j, state, x_struct, tree_ind, weight_std)
        {
            for (size_t i = 0; i < state->residual_std[0].size(); i++)
            {
                loglike_weight[j] += loglike_x(state, x_struct, tree_ind, i, weight_std[j]);
            }
        }
    }
    #pragma omp taskwait

    // Draw weight
    for (size_t i = 0; i < weight_std.size(); i++)
    {
        // loglike_weight[i] = weight_std[i] * loglike_pi + lgamma(weight_std[i] * n + 1) - lgamma(n + 1) - lgamma((weight_std[i] - 1) * n + 1);
        // loglike_weight[i] = weight_std[i] * loglike_pi - loglike_weight[i];
        if (loglike_weight[i] > max_loglike_weight){max_loglike_weight = loglike_weight[i];}
    }
    for (size_t i = 0; i < weight_std.size(); i++)
    {
        loglike_weight[i] = exp(loglike_weight[i] - max_loglike_weight);
    }

    std::discrete_distribution<> d(loglike_weight.begin(), loglike_weight.end());
    weight = weight_std[d(state->gen)];
    // cout << "weight " << weight << endl;


    // Draw phi
    std::gamma_distribution<double> gammadist(1.0, 1.0);

    for (size_t i = 0; i < state->residual_std[0].size(); i++)
    {
        for (size_t j = 0; j < dim_theta; ++j)
        {
            fits_w[j] = pow(state->residual_std[j][i] * (*(x_struct->data_pointers[tree_ind][i]))[j], weight);
        }

        (*phi)[i] = gammadist(state->gen) / (1.0 * accumulate(fits_w.begin(), fits_w.end(), 0.0) );
        if (isinf((*phi)[i])) {
            cout << "current weight " << weight << endl;
            terminate();
        }
    }


    return;
}

void LogitModel::initialize_root_suffstat(std::unique_ptr<State> &state, std::vector<double> &suff_stat)
{

    /*
    // sum of y
    suff_stat[0] = sum_vec(state->residual_std[0]);
    // sum of y squared
    suff_stat[1] = sum_squared(state->residual_std[0]);
    // number of observations in the node
    suff_stat[2] = state->n_y;
    */

    // JINGYU check -- should i always plan to resize this vector?
    // reply: use it for now. Not sure how to call constructor of tree when initialize vector<vector<tree>>, see definition of trees2 in XBART_multinomial, train_all.cpp

    // remove resizing it does not work, strange

    suff_stat.resize(2 * dim_theta);
    std::fill(suff_stat.begin(), suff_stat.end(), 0.0);
    for (size_t i = 0; i < state->n_y; i++)
    {
        // from 0
        incSuffStat(state->residual_std, i, suff_stat);
    }

    return;
}

void LogitModel::updateNodeSuffStat(std::vector<double> &suff_stat, matrix<double> &residual_std, matrix<size_t> &Xorder_std, size_t &split_var, size_t row_ind)
{
    /*
    suff_stat[0] += residual_std[0][Xorder_std[split_var][row_ind]];
    suff_stat[1] += pow(residual_std[0][Xorder_std[split_var][row_ind]], 2);
    suff_stat[2] += 1;
    */

    incSuffStat(residual_std, Xorder_std[split_var][row_ind], suff_stat);

    return;
}

void LogitModel::calculateOtherSideSuffStat(std::vector<double> &parent_suff_stat, std::vector<double> &lchild_suff_stat, std::vector<double> &rchild_suff_stat, size_t &N_parent, size_t &N_left, size_t &N_right, bool &compute_left_side)
{

    // in function split_xorder_std_categorical, for efficiency, the function only calculates suff stat of ONE child
    // this function calculate the other side based on parent and the other child

    if (compute_left_side)
    {
        rchild_suff_stat = parent_suff_stat - lchild_suff_stat;
        // this is a wired bug. Should look into it in the future.
        for (size_t j = 0; j < rchild_suff_stat.size() ; j++)
        {
            rchild_suff_stat[j] = rchild_suff_stat[j] > 0 ? rchild_suff_stat[j] : 0;
            // if (isnan(rchild_suff_stat[j])) {cout <<"j = " << j <<  "; parent_suff_stat = " << parent_suff_stat[j] << "; lchild_suff_stat = " << lchild_suff_stat[j]<< "; rchild_suff_stat = " << rchild_suff_stat[j]  << endl;}
            
        }
    }
    else
    {
        lchild_suff_stat = parent_suff_stat - rchild_suff_stat;
        for (size_t j = 0; j < rchild_suff_stat.size() ; j++)
        {
            lchild_suff_stat[j] = lchild_suff_stat[j] > 0 ? lchild_suff_stat[j] : 0;
            // if (isnan(lchild_suff_stat[j])) { cout <<"j = " << j <<  "; parent_suff_stat = " << parent_suff_stat[j] << "; rchild_suff_stat = " << rchild_suff_stat[j]<< "; lchild_suff_stat = " << lchild_suff_stat[j]  << endl;}
            
        }
    }
    return;
}

void LogitModel::state_sweep(size_t tree_ind, size_t M, matrix<double> &residual_std, std::unique_ptr<X_struct> &x_struct) const
{

    size_t next_index = tree_ind + 1;
    if (next_index == M)
    {
        next_index = 0;
    }

    ////////////////////////////////////////////////////////
    // Be care of line 151 in train_all.cpp, initial_theta
    ////////////////////////////////////////////////////////

    // cumulative product of trees, multiply current one, divide by next one

    for (size_t i = 0; i < residual_std[0].size(); i++)
    {
        for (size_t j = 0; j < dim_theta; ++j)
        {
            residual_std[j][i] = residual_std[j][i] * (*(x_struct->data_pointers[tree_ind][i]))[j] / (*(x_struct->data_pointers[next_index][i]))[j];
        }
    }

    return;
}

double LogitModel::likelihood(std::vector<double> &temp_suff_stat, std::vector<double> &suff_stat_all, size_t N_left, bool left_side, bool no_split, std::unique_ptr<State> &state) const
{
    // likelihood equation,
    // note the difference of left_side == true / false
    // node_suff_stat is mean of y, sum of square of y, saved in tree class
    // double y_sum = (double)suff_stat_all[2] * suff_stat_all[0];
    // double y_sum = suff_stat_all[0];
    // double suff_one_side;

    /////////////////////////////////////////////////////////////////////////
    //
    //  I know combining likelihood and likelihood_no_split looks nicer
    //  but this is a very fundamental function, executed many times
    //  the extra if(no_split) statement and value assignment make the code about 5% slower!!
    //
    /////////////////////////////////////////////////////////////////////////

    //could rewrite without all these local assigments if that helps...
    std::vector<double> local_suff_stat = suff_stat_all; // no split

    //COUT << "LIK" << endl;

    //COUT << "all suff stat dim " << suff_stat_all.size();

    if (!no_split)
    {
        if (left_side)
        {
            //COUT << "LEFTWARD HO" << endl;
            //COUT << "local suff stat dim " << local_suff_stat.size() << endl;
            //COUT << "temp suff stat dim " << temp_suff_stat.size() << endl;
            local_suff_stat = temp_suff_stat;
        }
        else
        {
            //COUT << "RIGHT HO" << endl;
            //COUT << "local suff stat dim " << local_suff_stat.size() << endl;
            //COUT << "temp suff stat dim " << temp_suff_stat.size() << endl;
            local_suff_stat = suff_stat_all - temp_suff_stat;
            for (size_t j = 0; j < local_suff_stat.size() ; j++)
            {
                // if (local_suff_stat[j] < 0){cout << "j = " << j << "; parrent suff = " << suff_stat_model[j] << "; left suff = " << temp_suff_stat[j] << endl;}
                local_suff_stat[j] = local_suff_stat[j] > 0 ? local_suff_stat[j] : 0;
            }
            // ntau = (suff_stat_all[2] - N_left - 1) * tau;
            // suff_one_side = y_sum - temp_suff_stat[0];
        }
    }

    // return 0.5 * log(sigma2) - 0.5 * log(nbtau + sigma2) + 0.5 * tau * pow(y_sum, 2) / (sigma2 * (nbtau + sigma2));

    //return - 0.5 * nb * log(2 * 3.141592653) -  0.5 * nb * log(sigma2) + 0.5 * log(sigma2) - 0.5 * log(nbtau + sigma2) - 0.5 * y_squared_sum / sigma2 + 0.5 * tau * pow(y_sum, 2) / (sigma2 * (nbtau + sigma2));

    return (LogitLIL(local_suff_stat));
}

void LogitModel::ini_residual_std(std::unique_ptr<State> &state)
{
    //double value = state->ini_var_yhat * ((double)state->num_trees - 1.0) / (double)state->num_trees;
    for (size_t i = 0; i < state->residual_std[0].size(); i++)
    {
        // init leaf pars are all 1, partial fits are all 1
        for (size_t j = 0; j < dim_theta; ++j)
        {
            state->residual_std[j][i] = 1.0; // (*state->y_std)[i] - value;
        }
    }
    return;
}

void LogitModel::predict_std(const double *Xtestpointer, size_t N_test, size_t p, size_t num_trees, size_t num_sweeps, matrix<double> &yhats_test_xinfo, vector<vector<tree>> &trees, std::vector<double> &output_vec)
{

    // output is a 3D array (armadillo cube), nsweeps by n by number of categories

    tree::tree_p bn;

    for (size_t sweeps = 0; sweeps < num_sweeps; sweeps++)
    {

        for (size_t data_ind = 0; data_ind < N_test; data_ind++)
        {

            for (size_t i = 0; i < trees[0].size(); i++)
            {
                // search leaf
                bn = trees[sweeps][i].search_bottom_std(Xtestpointer, data_ind, p, N_test);

                for (size_t k = 0; k < dim_residual; k++)
                {
                    // add all trees

                    // product of trees, thus sum of logs

                    output_vec[sweeps + data_ind * num_sweeps + k * num_sweeps * N_test] += bn->weight * log(bn->theta_vector[k]);
                }
            }
        }
    }

    // normalizing probability

    double denom = 0.0;
    double max_log_prob = -INFINITY;

    for (size_t sweeps = 0; sweeps < num_sweeps; sweeps++)
    {
        for (size_t data_ind = 0; data_ind < N_test; data_ind++)
        {

            max_log_prob = -INFINITY;
            // take exp, subtract max to avoid overflow

            // this line does not work for some reason, havd to write loops manually
            // output.tube(sweeps, data_ind) = exp(output.tube(sweeps, data_ind) - output.tube(sweeps, data_ind).max());

            // find max of probability for all classes
            for (size_t k = 0; k < dim_residual; k++)
            {
                if(output_vec[sweeps + data_ind * num_sweeps + k * num_sweeps * N_test] > max_log_prob){
                    max_log_prob = output_vec[sweeps + data_ind * num_sweeps + k * num_sweeps * N_test];
                }
            }

            // take exp after subtracting max to avoid overflow
            for (size_t k = 0; k < dim_residual; k++)
            {
                output_vec[sweeps + data_ind * num_sweeps + k * num_sweeps * N_test] = exp(output_vec[sweeps + data_ind * num_sweeps + k * num_sweeps * N_test] - max_log_prob);
            }

            // calculate normalizing constant
            denom = 0.0;
            for (size_t k = 0; k < dim_residual; k++)
            {
                // denom += output_vec[sweeps + data_ind * num_sweeps + k * num_sweeps * N_test];
                denom += output_vec[sweeps + data_ind * num_sweeps + k * num_sweeps * N_test];
            }

            // normalizing
            for (size_t k = 0; k < dim_residual; k++)
            {
                output_vec[sweeps + data_ind * num_sweeps + k * num_sweeps * N_test] = output_vec[sweeps + data_ind * num_sweeps + k * num_sweeps * N_test] / denom;
            }
        }
    }
    return;
}

// this function is for a standalone prediction function for classification case.
// with extra input iteration, which specifies which iteration (sweep / forest) to use
void LogitModel::predict_std_standalone(const double *Xtestpointer, size_t N_test, size_t p, size_t num_trees, size_t num_sweeps, matrix<double> &yhats_test_xinfo, vector<vector<tree>> &trees, std::vector<double> &output_vec, std::vector<size_t>& iteration, double weight)
{

    // output is a 3D array (armadillo cube), nsweeps by n by number of categories

    size_t num_iterations = iteration.size();

    tree::tree_p bn;

    cout << "number of iterations " << num_iterations << " " << num_sweeps << endl;

    size_t sweeps;

    for (size_t iter = 0; iter < num_iterations; iter++)
    {
        sweeps = iteration[iter];

        for (size_t data_ind = 0; data_ind < N_test; data_ind++)
        {

            for (size_t i = 0; i < trees[0].size(); i++)
            {
                // search leaf
                bn = trees[sweeps][i].search_bottom_std(Xtestpointer, data_ind, p, N_test);

                for (size_t k = 0; k < dim_residual; k++)
                {
                    // add all trees

                    // product of trees, thus sum of logs

                    output_vec[iter + data_ind * num_iterations + k * num_iterations * N_test] += bn->weight * log(bn->theta_vector[k]);
                }
            }
        }
    }
    // normalizing probability

    double denom = 0.0;
    double max_log_prob = -INFINITY;

    for (size_t iter = 0; iter < num_iterations; iter++)
    {

        sweeps = iteration[iter];

        for (size_t data_ind = 0; data_ind < N_test; data_ind++)
        {

            max_log_prob = -INFINITY;
            // take exp, subtract max to avoid overflow

            // this line does not work for some reason, havd to write loops manually
            // output.tube(sweeps, data_ind) = exp(output.tube(sweeps, data_ind) - output.tube(sweeps, data_ind).max());

            // find max of probability for all classes
            for (size_t k = 0; k < dim_residual; k++)
            {
                if(output_vec[iter + data_ind * num_iterations + k * num_iterations * N_test] > max_log_prob){
                    max_log_prob = output_vec[iter + data_ind * num_iterations + k * num_iterations * N_test];
                }
            }

            // take exp after subtracting max to avoid overflow
            for (size_t k = 0; k < dim_residual; k++)
            {
                output_vec[iter + data_ind * num_iterations + k * num_iterations * N_test] = exp(output_vec[iter + data_ind * num_iterations + k * num_iterations * N_test] - max_log_prob);
            }

            // calculate normalizing constant
            denom = 0.0;
            for (size_t k = 0; k < dim_residual; k++)
            {
                // denom += output_vec[iter + data_ind * num_iterations + k * num_iterations * N_test];
                denom +=  output_vec[iter + data_ind * num_iterations + k * num_iterations * N_test] ;
            }

            // normalizing
            for (size_t k = 0; k < dim_residual; k++)
            {
                output_vec[iter + data_ind * num_iterations + k * num_iterations * N_test] = output_vec[iter + data_ind * num_iterations + k * num_iterations * N_test] / denom;
            }
        }
    }
    return;
}



//////////////////////////////////////////////////////////////////////////////////////
//
//
//  Logit Model Separate Trees
//
//
//
//////////////////////////////////////////////////////////////////////////////////////

//incSuffStat should take a state as its first argument
void LogitModelSeparateTrees::incSuffStat(matrix<double> &residual_std, size_t index_next_obs, std::vector<double> &suffstats)
{
    suffstats[(*y_size_t)[index_next_obs]] += 1; //weight;

    size_t j = class_operating;
    suffstats[dim_theta + j] += (*phi)[index_next_obs] * pow(residual_std[j][index_next_obs], weight);

    return;
}

void LogitModelSeparateTrees::samplePars(std::unique_ptr<State> &state, std::vector<double> &suff_stat, std::vector<double> &theta_vector, double &prob_leaf, const size_t &tree_ind)
{
    
    size_t c = dim_residual;
    size_t j = class_operating;
    if (weight == 1 | suff_stat[c + j] > tau_b + 500)
    {
        std::gamma_distribution<double> gammadist(tau_a + suff_stat[j] +1/weight, 1.0);

        theta_vector[j] =  pow(gammadist(state->gen) / (tau_b + suff_stat[c + j]), 1 / weight ) ;
    }
    else if (suff_stat[c + j] == 0)
    {
        std::gamma_distribution<double> gammadist(tau_a + suff_stat[j] * weight);
        theta_vector[j] = gammadist(state->gen) / (tau_b);
    }
    else
    {
        LogitParams *lparams = new LogitParams(tau_a, tau_b, weight, suff_stat[j], suff_stat[c + j]);

        double mx, output;
        int status_mx = get_root(derive_logit_kernel, lparams, mx,  5.0, 10000, 1e-6); // status_mx = 1 if can't find root
        if ( !status_mx) 
        { 
            // lparams->set_mx(mx);
            lparams->set_logv(log_logit_kernel(mx, lparams));
        } // set logv if find root
        else
        {
            lparams->print();
            cout << "can't find root in samplePars, use general Gamma"  << endl;   
            // need to reconsider, how to sample when we can't find root??
    
            std::gamma_distribution<double> gammadist(tau_a + suff_stat[j] +1/weight, 1.0);

            theta_vector[j] =  pow(gammadist(state->gen) / (tau_b + suff_stat[c + j]), 1 / weight ) ;

            return;
        }
            
        int status = get_integration(LogitKernel, lparams, output, 0.5*mx);

        if (output == 0)
        {
            lparams->print();
            cout << "integration = 0 in samplePars, use general Gamma" << endl;
            std::gamma_distribution<double> gammadist(tau_a + suff_stat[j] +1/weight, 1.0);

            theta_vector[j] =  pow(gammadist(state->gen) / (tau_b + suff_stat[c + j]), 1 / weight ) ;

            return;
        }
        else if( status)
        {
            lparams->print();
            cout << "integration failed in samplePars. output = " << output << ", use general Gamma" << endl;
    
            std::gamma_distribution<double> gammadist(tau_a + suff_stat[j] +1/weight, 1.0);

            theta_vector[j] =  pow(gammadist(state->gen) / (tau_b + suff_stat[c + j]), 1 / weight ) ;
            
            return;
        }

        double logk = log(output) + lparams->logv;

        double mval = LogitKernel(mx, lparams) / output;

        // double sigma = 1 / sqrt(weight * suff_stat[c+j] + tau_b);
        double sigma = 1 / weight / sqrt(suff_stat[c+j] + tau_b);

        boost::math::lognormal_distribution<double> dlnorm(log(mx), sigma);
        std::lognormal_distribution<double> rlnorm(log(mx), sigma);
        std::uniform_real_distribution<double> runif(0.0,1.0);

        double M = mval / pdf(dlnorm, exp(log(mx))); 

        double theta, u;
        size_t count_reject = 0;
        size_t reject_limit = 100;
        while(count_reject < reject_limit)
        {
            theta = rlnorm(state->gen);
            u = runif(state->gen);
            // cout << "theta = " << theta << "; u = " << u << ";" << endl;
            // cout << "logf = " << log_logit_kernel(theta, lparams) << "; log(k) = " << log(k) << "; log dlnorm = " << log(pdf(dlnorm, theta)) << "; log(M) = " << log(M) << endl;
            if (u < exp(log_logit_kernel(theta, lparams) - lparams->logv - logk - log(pdf(dlnorm, theta)) - log(M)))
            {
                theta_vector[j] = theta;
                return;
            }
            count_reject += 1;
        }
        
        lparams->print();
        cout << "warning: reject sampling after " <<  reject_limit << " iterations, use general Gamma" << endl;
                
        std::gamma_distribution<double> gammadist(tau_a + suff_stat[j] +1/weight, 1.0);
        theta_vector[j] =  pow(gammadist(state->gen) / (tau_b + suff_stat[c + j]), 1 / weight ) ;
    }
 
    return;
}

void LogitModelSeparateTrees::update_state(std::unique_ptr<State> &state, size_t tree_ind, std::unique_ptr<X_struct> &x_struct)
{

    double sum_fits = 0;
    double loglike_pi = 0;
    size_t y_i;
    double sum_log_fits;
    

    std::gamma_distribution<double> gammadist(1.0, 1.0);
    std::vector<double> fits_w(dim_residual, 0.0);

    // Draw phi
    double temp;
    for (size_t i = 0; i < state->residual_std[0].size(); i++){
        for (size_t j = 0; j < dim_theta; ++j)
        {
            fits_w[j] = pow(state->residual_std[j][i] * (*(x_struct->data_pointers_multinomial[j][tree_ind][i]))[j], weight);
        }
        // (*phi)[i] = gammadist(state->gen) / (1.0*sum_fits_v[i]/min_fits); 
        temp = gammadist(state->gen) / (1.0 * accumulate(fits_w.begin(), fits_w.end(), 0.0) );
        // cout <<"phi_" << i << ": " << temp <<endl;
        (*phi)[i] = temp;
        if (isinf(temp)) {
            cout << "current weight " << weight << endl;
            for (size_t j = 0; j < dim_theta; ++j)
            {
                fits_w[j] = state->residual_std[j][i] * (*(x_struct->data_pointers_multinomial[j][tree_ind][i]))[j];
            }
            cout << "fits " << fits_w << endl;
            terminate();
        }
    }

    // if (tree_ind == state->num_trees - 1) // update weight after last tree
    // {
    min_fits = INFINITY;
    double max_loglike_weight = -INFINITY;
    // double max_logf = -INFINITY;
    std::vector<double> log_f(dim_residual, 0.0);
    // std::vector<double> sum_fits_v (state->residual_std[0].size(), 0.0);
    std::vector<double> sum_fits_weight(weight_std.size(), 0.0);
    std::vector<double> loglike_weight(weight_std.size(), 0.0);
    
    
    std::vector<double> log_lambda_prior(weight_std.size(), 0.0);


    for (size_t j = 0; j < weight_std.size(); j++)
    {
        #pragma omp task shared(loglike_weight, j, state, x_struct, tree_ind, weight_std, dim_residual, log_lambda_prior)
        {
            for (size_t i = 0; i < state->residual_std[0].size(); i++)
            {
                loglike_weight[j] += loglike_x(state, x_struct, tree_ind, i, weight_std[j]);

                // for (size_t k = 0; k < state->num_trees; k++)
                // {
                //     for (size_t l = 0; l < dim_residual; l++)
                //     {
                //         log_lambda_prior[j] += log_dlambda((*(x_struct->data_pointers_multinomial[l][k][i]))[l], weight_std[j]);
                //     }
                // }
            }

        }
    }
    #pragma omp taskwait
    // Draw weight

    for (size_t i = 0; i < weight_std.size(); i++)
    {
        // loglike_weight[i] = weight_std[i] * loglike_pi + lgamma(weight_std[i] * n + 1) - lgamma(n + 1) - lgamma((weight_std[i] - 1) * n + 1);
        // loglike_weight[i] = weight_std[i] * loglike_pi - loglike_weight[i];
        if (loglike_weight[i] + log_lambda_prior[i] > max_loglike_weight){max_loglike_weight = loglike_weight[i] + log_lambda_prior[i];}
    }
    for (size_t i = 0; i < weight_std.size(); i++)
    {
        loglike_weight[i] = exp(loglike_weight[i]  + log_lambda_prior[i] - max_loglike_weight);
    }
    // cout << "loglike_weight " << loglike_weight <<endl;
    
    std::discrete_distribution<> d(loglike_weight.begin(), loglike_weight.end());
    weight = weight_std[d(state->gen)];
    // cout << "weight " << weight << endl;
    // }

    return;
}

void LogitModelSeparateTrees::initialize_root_suffstat(std::unique_ptr<State> &state, std::vector<double> &suff_stat)
{

    /*
    // sum of y
    suff_stat[0] = sum_vec(state->residual_std[0]);
    // sum of y squared
    suff_stat[1] = sum_squared(state->residual_std[0]);
    // number of observations in the node
    suff_stat[2] = state->n_y;
    */

    // JINGYU check -- should i always plan to resize this vector?
    // reply: use it for now. Not sure how to call constructor of tree when initialize vector<vector<tree>>, see definition of trees2 in XBART_multinomial, train_all.cpp

    // remove resizing it does not work, strange

    suff_stat.resize(2 * dim_theta);
    std::fill(suff_stat.begin(), suff_stat.end(), 0.0);
    for (size_t i = 0; i < state->n_y; i++)
    {
        // from 0
        incSuffStat(state->residual_std, i, suff_stat);
    }

    return;
}

void LogitModelSeparateTrees::updateNodeSuffStat(std::vector<double> &suff_stat, matrix<double> &residual_std, matrix<size_t> &Xorder_std, size_t &split_var, size_t row_ind)
{
    /*
    suff_stat[0] += residual_std[0][Xorder_std[split_var][row_ind]];
    suff_stat[1] += pow(residual_std[0][Xorder_std[split_var][row_ind]], 2);
    suff_stat[2] += 1;
    */

    incSuffStat(residual_std, Xorder_std[split_var][row_ind], suff_stat);

    return;
}

void LogitModelSeparateTrees::calculateOtherSideSuffStat(std::vector<double> &parent_suff_stat, std::vector<double> &lchild_suff_stat, std::vector<double> &rchild_suff_stat, size_t &N_parent, size_t &N_left, size_t &N_right, bool &compute_left_side)
{

    // in function split_xorder_std_categorical, for efficiency, the function only calculates suff stat of ONE child
    // this function calculate the other side based on parent and the other child

    if (compute_left_side)
    {
        rchild_suff_stat = parent_suff_stat - lchild_suff_stat;
        // this is a wired bug. Should look into it in the future.
        for (size_t j = 0; j < rchild_suff_stat.size() ; j++)
        {
            rchild_suff_stat[j] = rchild_suff_stat[j] > 0 ? rchild_suff_stat[j] : 0;
            // if (isnan(rchild_suff_stat[j])) {cout <<"j = " << j <<  "; parent_suff_stat = " << parent_suff_stat[j] << "; lchild_suff_stat = " << lchild_suff_stat[j]<< "; rchild_suff_stat = " << rchild_suff_stat[j]  << endl;}
            
        }
    }
    else
    {
        lchild_suff_stat = parent_suff_stat - rchild_suff_stat;
        for (size_t j = 0; j < rchild_suff_stat.size() ; j++)
        {
            lchild_suff_stat[j] = lchild_suff_stat[j] > 0 ? lchild_suff_stat[j] : 0;
            // if (isnan(lchild_suff_stat[j])) { cout <<"j = " << j <<  "; parent_suff_stat = " << parent_suff_stat[j] << "; rchild_suff_stat = " << rchild_suff_stat[j]<< "; lchild_suff_stat = " << lchild_suff_stat[j]  << endl;}
            
        }
    }
    return;
}


void LogitModelSeparateTrees::state_sweep(size_t tree_ind, size_t M, matrix<double> &residual_std, std::unique_ptr<X_struct> &x_struct) const
{

    size_t next_index = tree_ind + 1;
    if (next_index == M)
    {
        next_index = 0;
    }

    ////////////////////////////////////////////////////////
    // Be care of line 151 in train_all.cpp, initial_theta
    ////////////////////////////////////////////////////////

    // cumulative product of trees, multiply current one, divide by next one

    for (size_t i = 0; i < residual_std[0].size(); i++)
    {
        for (size_t j = 0; j < dim_theta; ++j)
        {
            residual_std[j][i] = residual_std[j][i] * (*(x_struct->data_pointers_multinomial[j][tree_ind][i]))[j] / (*(x_struct->data_pointers_multinomial[j][next_index][i]))[j];
        }
    }

    return;
}

double LogitModelSeparateTrees::likelihood(std::vector<double> &temp_suff_stat, std::vector<double> &suff_stat_all, size_t N_left, bool left_side, bool no_split, std::unique_ptr<State> &state) const
{
    // likelihood equation,
    // note the difference of left_side == true / false
    // node_suff_stat is mean of y, sum of square of y, saved in tree class
    // double y_sum = (double)suff_stat_all[2] * suff_stat_all[0];
    // double y_sum = suff_stat_all[0];
    // double suff_one_side;

    /////////////////////////////////////////////////////////////////////////
    //
    //  I know combining likelihood and likelihood_no_split looks nicer
    //  but this is a very fundamental function, executed many times
    //  the extra if(no_split) statement and value assignment make the code about 5% slower!!
    //
    /////////////////////////////////////////////////////////////////////////

    //could rewrite without all these local assigments if that helps...
    std::vector<double> local_suff_stat = suff_stat_all; // no split

    //COUT << "LIK" << endl;

    //COUT << "all suff stat dim " << suff_stat_all.size();

    if (!no_split)
    {
        if (left_side)
        {
            //COUT << "LEFTWARD HO" << endl;
            //COUT << "local suff stat dim " << local_suff_stat.size() << endl;
            //COUT << "temp suff stat dim " << temp_suff_stat.size() << endl;
            local_suff_stat = temp_suff_stat;
        }
        else
        {
            //COUT << "RIGHT HO" << endl;
            //COUT << "local suff stat dim " << local_suff_stat.size() << endl;
            //COUT << "temp suff stat dim " << temp_suff_stat.size() << endl;
            local_suff_stat = suff_stat_all - temp_suff_stat;
            for (size_t j = 0; j < local_suff_stat.size() ; j++)
            {
                // if (local_suff_stat[j] < 0){cout << "j = " << j << "; parrent suff = " << suff_stat_model[j] << "; left suff = " << temp_suff_stat[j] << endl;}
                local_suff_stat[j] = local_suff_stat[j] > 0 ? local_suff_stat[j] : 0;
            }
            // ntau = (suff_stat_all[2] - N_left - 1) * tau;
            // suff_one_side = y_sum - temp_suff_stat[0];
        }
    }

    // return 0.5 * log(sigma2) - 0.5 * log(nbtau + sigma2) + 0.5 * tau * pow(y_sum, 2) / (sigma2 * (nbtau + sigma2));

    //return - 0.5 * nb * log(2 * 3.141592653) -  0.5 * nb * log(sigma2) + 0.5 * log(sigma2) - 0.5 * log(nbtau + sigma2) - 0.5 * y_squared_sum / sigma2 + 0.5 * tau * pow(y_sum, 2) / (sigma2 * (nbtau + sigma2));

    return (LogitLIL(local_suff_stat));
}

void LogitModelSeparateTrees::ini_residual_std(std::unique_ptr<State> &state)
{
    //double value = state->ini_var_yhat * ((double)state->num_trees - 1.0) / (double)state->num_trees;
    for (size_t i = 0; i < state->residual_std[0].size(); i++)
    {
        // init leaf pars are all 1, partial fits are all 1
        for (size_t j = 0; j < dim_theta; ++j)
        {
            state->residual_std[j][i] = 1.0; // (*state->y_std)[i] - value;
        }
    }
    return;
}

void LogitModelSeparateTrees::predict_std(const double *Xtestpointer, size_t N_test, size_t p, size_t num_trees, size_t num_sweeps, matrix<double> &yhats_test_xinfo, vector<vector<vector<tree>>> &trees, std::vector<double> &output_vec)
{

    // output is a 3D array (armadillo cube), nsweeps by n by number of categories

    tree::tree_p bn;

    for (size_t data_ind = 0; data_ind < N_test; data_ind++)
    { // for each data observation

        for (size_t sweeps = 0; sweeps < num_sweeps; sweeps++)
        {

            for (size_t k = 0; k < dim_residual; k++)
            { // loop over class

                for (size_t i = 0; i < trees[0][0].size(); i++)
                {
                    bn = trees[k][sweeps][i].search_bottom_std(Xtestpointer, data_ind, p, N_test);

                    // product of trees, thus sum of logs

                    // cout << "one obs " << log(bn->theta_vector[k]) << "  "  << bn->theta_vector[k]  << endl;

                    if (bn->weight < 1){
                        cout << "class " << k << ", sweep " << sweeps << ", tree " << i << ", leaf weight = " << bn->weight << endl;
                        bn->weight = 5; 
                    }

                    output_vec[sweeps + data_ind * num_sweeps + k * num_sweeps * N_test] += bn->weight * log(bn->theta_vector[k]); // need to powered by weight!!
                }
            }
        }
    }

    // normalizing probability

    double denom = 0.0;
    double max_log_prob = -INFINITY;

    for (size_t sweeps = 0; sweeps < num_sweeps; sweeps++)
    {
        for (size_t data_ind = 0; data_ind < N_test; data_ind++)
        {

            max_log_prob = -INFINITY;
            // take exp, subtract max to avoid overflow

            // this line does not work for some reason, havd to write loops manually
            // output.tube(sweeps, data_ind) = exp(output.tube(sweeps, data_ind) - output.tube(sweeps, data_ind).max());

            // find max of probability for all classes
            for (size_t k = 0; k < dim_residual; k++)
            {
                if(output_vec[sweeps + data_ind * num_sweeps + k * num_sweeps * N_test] > max_log_prob){
                    max_log_prob = output_vec[sweeps + data_ind * num_sweeps + k * num_sweeps * N_test];
                }
            }

            // take exp after subtracting max to avoid overflow
            for (size_t k = 0; k < dim_residual; k++)
            {
                output_vec[sweeps + data_ind * num_sweeps + k * num_sweeps * N_test] = exp(output_vec[sweeps + data_ind * num_sweeps + k * num_sweeps * N_test] - max_log_prob);
            }

            // calculate normalizing constant
            denom = 0.0;
            for (size_t k = 0; k < dim_residual; k++)
            {
                // denom += output_vec[sweeps + data_ind * num_sweeps + k * num_sweeps * N_test];
                denom += output_vec[sweeps + data_ind * num_sweeps + k * num_sweeps * N_test];
            }

            // normalizing
            for (size_t k = 0; k < dim_residual; k++)
            {
                output_vec[sweeps + data_ind * num_sweeps + k * num_sweeps * N_test] = output_vec[sweeps + data_ind * num_sweeps + k * num_sweeps * N_test] / denom;
            }
        }
    }
    return;
}

// this function is for a standalone prediction function for classification case.
// with extra input iteration, which specifies which iteration (sweep / forest) to use
void LogitModelSeparateTrees::predict_std_standalone(const double *Xtestpointer, size_t N_test, size_t p, size_t num_trees, size_t num_sweeps, matrix<double> &yhats_test_xinfo,  vector<vector<vector<tree>>> &trees, std::vector<double> &output_vec, std::vector<size_t>& iteration, double weight)
{

    // output is a 3D array (armadillo cube), nsweeps by n by number of categories

    size_t num_iterations = iteration.size();

    tree::tree_p bn;

    cout << "number of iterations " << num_iterations << " " << num_sweeps << endl;

    size_t sweeps;

    for (size_t iter = 0; iter < num_iterations; iter++)
    {
        sweeps = iteration[iter];

        for (size_t data_ind = 0; data_ind < N_test; data_ind++)
        {

            for (size_t i = 0; i < trees[0][0].size(); i++)
            {
                
                for (size_t k = 0; k < dim_residual; k++)
                {
                    // search leaf
                    bn = trees[k][sweeps][i].search_bottom_std(Xtestpointer, data_ind, p, N_test);

                    // add all trees

                    // product of trees, thus sum of logs

                    output_vec[iter + data_ind * num_iterations + k * num_iterations * N_test] += bn->weight * log(bn->theta_vector[k]);// need to powered by weight!!
                }
            }
        }
    }
    
    // normalizing probability

    double denom = 0.0;
    double max_log_prob = -INFINITY;

    for (size_t iter = 0; iter < num_iterations; iter++)
    {

        sweeps = iteration[iter];

        for (size_t data_ind = 0; data_ind < N_test; data_ind++)
        {

            max_log_prob = -INFINITY;
            // take exp, subtract max to avoid overflow

            // this line does not work for some reason, havd to write loops manually
            // output.tube(sweeps, data_ind) = exp(output.tube(sweeps, data_ind) - output.tube(sweeps, data_ind).max());

            // find max of probability for all classes
            for (size_t k = 0; k < dim_residual; k++)
            {
                if(output_vec[iter + data_ind * num_iterations + k * num_iterations * N_test] > max_log_prob){
                    max_log_prob = output_vec[iter + data_ind * num_iterations + k * num_iterations * N_test];
                }
            }

            // take exp after subtracting max to avoid overflow
            for (size_t k = 0; k < dim_residual; k++)
            {
                output_vec[iter + data_ind * num_iterations + k * num_iterations * N_test] = exp(output_vec[iter + data_ind * num_iterations + k * num_iterations * N_test] - max_log_prob);
            }

            // calculate normalizing constant
            denom = 0.0;
            for (size_t k = 0; k < dim_residual; k++)
            {
                // denom += output_vec[iter + data_ind * num_iterations + k * num_iterations * N_test];
                denom +=  output_vec[iter + data_ind * num_iterations + k * num_iterations * N_test] ;
            }

            // normalizing
            for (size_t k = 0; k < dim_residual; k++)
            {
                output_vec[iter + data_ind * num_iterations + k * num_iterations * N_test] = output_vec[iter + data_ind * num_iterations + k * num_iterations * N_test] / denom;
            }
        }
    }
    return;
}

//////////////////////////////////////////////////////////////////////////////////////
//
//
//  Probit Model
//
//
//////////////////////////////////////////////////////////////////////////////////////
void ProbitClass::update_state(std::unique_ptr<State> &state, size_t tree_ind, std::unique_ptr<X_struct> &x_struct)
{
    // Draw Sigma
    // state->residual_std_full = state->residual_std - state->predictions_std[tree_ind];

    // residual_std is only 1 dimensional for regression model

    // std::vector<double> full_residual(state->n_y);

    // for (size_t i = 0; i < state->residual_std[0].size(); i++)
    // {
    //     full_residual[i] = state->residual_std[0][i] - (*(x_struct->data_pointers[tree_ind][i]))[0];
    // }

    // For probit model, do not need to sample gamma
    // std::gamma_distribution<double> gamma_samp((state->n_y + kap) / 2.0, 2.0 / (sum_squared(full_residual) + s));
    // state->update_sigma(1.0 / sqrt(gamma_samp(state->gen)));

    //update latent variable Z

    z_prev = z;

    double mu_temp;
    double u;

    for (size_t i = 0; i < state->n_y; i++)
    {
        a = 0;
        b = 1;

        mu_temp = normCDF(z_prev[i]);

        // Draw from truncated normal via inverse CDF methods
        if ((*state->y_std)[i] > 0)
        {
            a = std::min(mu_temp, 0.999);
        }
        else
        {
            b = std::max(mu_temp, 0.001);
        }

        std::uniform_real_distribution<double> unif(a, b);
        u = unif(state->gen);
        z[i] = normCDFInv(u) + mu_temp;
    }
    return;

    //NormalModel::update_state(state, tree_ind, x_struct);
}

void ProbitClass::state_sweep(size_t tree_ind, size_t M, matrix<double> &residual_std, std::unique_ptr<X_struct> &x_struct) const
{

    NormalModel::state_sweep(tree_ind, M, residual_std, x_struct);
    size_t next_index = tree_ind + 1;
    if (next_index == M)
    {
        next_index = 0;
    }

    ////////////////////////////////////////////////////////
    // Be care of line 151 in train_all.cpp, initial_theta
    ////////////////////////////////////////////////////////

    for (size_t i = 0; i < residual_std[0].size(); i++)
    {
        residual_std[0][i] = residual_std[0][i] - z_prev[i] + z[i];
    }

    return;
}

//////////////////////////////////////////////////////////////////////////////////////
//
//
//  CLT Model
//
//
//////////////////////////////////////////////////////////////////////////////////////
