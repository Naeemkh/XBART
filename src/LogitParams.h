#ifndef GUARD_LogitParams_h
#define GUARD_LogitParams_h
#include "common.h"

using namespace std;


struct LogitParams{


public:
    double tau_a, tau_b, w, r, s;

    LogitParams(): tau_a(1.0), tau_b(1.0), w(1), r(0.0), s(0.0) {}

    LogitParams(double tau_a, double tau_b, double w, double suff_r, double suff_s): tau_a(tau_a), tau_b(tau_b), w(w), r(suff_r), s(suff_s) {}

    void print()
    {
        cout << "w = " << this->w << "; r = " << this->r << "; s = " << this->s << endl;
        return; 
    }

    void set_w(double w) {this->w = w;}

    void set_r(double r) {this->r = r;}

    void set_s(double s) {this->s = s;}

    void set_params(double tau_a, double tau_b, double w, double suff_r, double suff_s)
    {
        this->tau_a = tau_a;
        this->tau_b = tau_b;
        this-> w = w;
        this-> r = suff_r;
        this-> s = suff_s;
        return;
    }
    
};

#endif