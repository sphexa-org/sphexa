/*
 * MIT License
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/*! @file
 *
 * @brief This routine produces 1d solutions for a sedov blast wave propagating through a density gradient: rho = rho**(-omega)
 *        , in planar(1D), cylindrical(2D) or spherical geometry(3D) for the 'standard', 'singular' and 'vaccum' cases.
 *
 *        - standard case: a nonzero solution extends from the shock to the origin,       where the pressure is finite.
 *        - singular case: a nonzero solution extends from the shock to the origin,       where the pressure vanishes.
 *        - vacuum case  : a nonzero solution extends from the shock to a boundary point, where the density vanishes making the pressure meaningless.
 *
 *        This routine is a C++ conversion of one Fortran code based in these two papers:
 *        - "evaluation of the sedov-von neumann-taylor blast wave solution", Jim Kamm, la-ur-00-6055
 *        - "the sedov self-similiar point blast solutions in nonuniform media", David Book, shock waves, 4, 1, 1994
 *
 *        Although the ordinary differential equations are analytic, the sedov expressions appear to become singular for various
 *        combinations of parameters and at the lower limits of the integration range.
 *        All these singularies are removable and done so by this routine.
 *
 * @author Jose A. Escartin <ja.escartin@gmail.com>
 *
 */

#pragma once

#include <cmath>
#include <vector>

#include <iostream>
#include <iomanip>

#include <fstream>

#include <string>

using namespace std;


class SedovAnalyticalSolution
{
public:

    static void create(const size_t dim,              // Dimensions
                       const double r0,               // Initial radio
                       const double r1,               // End radio
                       const size_t rPoints,          // Number of points between r0-r1
                       const double time,             // Time at solution
                       const double eblast,           // Energy blast in the wave front
                       const double omega_i,          // Energy blast in the wave front
                       const double gamma_i,          // Adiabatic coeficient
                       const double rho0,             // Initial density
                       const double u0,               // Initial internal energy
                       const double p0,               // Initial pressure
                       const double vel0,             // Initial velocity
                       const double cs0,              // Initial sound speed
                       const string outfile)          // Output solution filename
    {

        double rStep = (r1 - r0) / rPoints;
        double r[rPoints];
        for(size_t i = 0; i < rPoints; i++)
        {
            r[i] = r0 + (0.5 * rStep) + (i * rStep);
        }

        double rho[rPoints];
        double u[rPoints];
        double p[rPoints];
        double vel[rPoints];
        double cs[rPoints];

        sedovSol(dim, rPoints, time,
                 eblast, omega_i, gamma_i,
                 rho0, u0, p0, vel0, cs0,
                 r, rho, u, p, vel, cs);

        ofstream out(outfile);

        out <<        setw( 5) << "i"                 //
            << " " << setw(13) << "r[i]"              //
            << " " << setw(13) << "rho[i]/rho0"       //
            << " " << setw(13) << "u[i]"              //
            << " " << setw(13) << "p[i]"              //
            << " " << setw(13) << "vel[i]"            //
            << " " << setw(13) << "cs[i]"             //
            << endl;

        for(size_t i = 0; i < rPoints; i++)
        {
            if (i + 1 == 583)
            {
                cout << endl;
                cout << "i           = " << i + 1       << endl;
                cout << "r[i]        = " << r[i]        << endl;
                cout << "rho[i]/rho0 = " << rho[i]/rho0 << endl;
                cout << "u[i]        = " << u[i]        << endl;
                cout << "p[i]        = " << p[i]        << endl;
                cout << "vel[i]      = " << vel[i]      << endl;
                cout << "cs[i]       = " << cs[i]       << endl;
            }

            out <<        setw( 5) << i                                                 //
                << " " << setw(13) << setprecision(6) << std::scientific << r[i]        //
                << " " << setw(13) << setprecision(6) << std::scientific << rho[i]/rho0 //
                << " " << setw(13) << setprecision(6) << std::scientific << u[i]        //
                << " " << setw(13) << setprecision(6) << std::scientific << p[i]        //
                << " " << setw(13) << setprecision(6) << std::scientific << vel[i]      //
                << " " << setw(13) << setprecision(6) << std::scientific << cs[i]       //
                << endl;
        }
    }

private:

    // Constants
    static constexpr double eps    = 1.e-10;          //  eps controls the integration accuracy, don't get too greedy or the number of function evaluations required kills.
    static constexpr double eps2   = 1.e-30;          //  eps2 controls the root find accuracy
    static constexpr double osmall = 1.e-4;           //  osmall controls the size of transition regions

    // Global variables
    static double xgeom, omega, gamma;                //
    static double gamm1,gamp1, gpogm, xg2;            //
    static bool   lsingular,lstandard,lvacuum;        //
    static bool   lomega2,lomega3;                    //
    static double a0,a1,a2,a3,a4,a5;                  //
    static double a_val, b_val, c_val, d_val, e_val;  //
    static double rwant, vwant;                       //
    static double r2,    v0,vv,rvv;                   //
    static double gam_int;                            //


    static void sedovSol(const size_t dim,            // geometry factor: 1=planar, 2=cylindircal, 3=spherical
                         const size_t rPoints,        // Number of points between r0-r1
                         const double time,           // temporal point where solution is desired [seconds]
                         const double eblast,         // energy of blast in the wave front [erg]
                         const double omega_i,        // density power law exponent in 'rho = rho0 * r**(-omega)'
                         const double gamma_i,        // gamma law equation of state
                         const double rho0,           // ambient density g/cm**3 in 'rho = rho0 * r**(-omega)'
                         const double u0,             // ambient internal energy [erg/g]
                         const double p0,             // ambient pressure [erg/cm**3]
                         const double vel0,           // ambient material speed [cm/s]
                         const double cs0,            // ambient sound speed [cm/s]
                         double *     r,              // out: spatial points where solution is desired [cm]
                         double *     rho,            // out: density  [g/cm**3]
                         double *     u,              // out: specific internal energy [erg/g]
                         double *     p,              // out: presssure [erg/cm**3]
                         double *     vel,            // out: velocity [cm/s]
                         double *     cs)             // out: sound speed [cm/s]
    {
        // Local variables
        double eval1 = 0.;
        double eval2 = 0.;

        // Set global input parameters
        omega = omega_i;  //
        gamma = gamma_i;  //
        xgeom = dim;      //

        // Check unphysical cases
        if (omega >= xgeom)
        {
            cout << "Unphysical case: Infinite mass" << endl;
            return;
        }

        // Frequest combination variables
        gamm1 = gamma - 1.;                                      //
        gamp1 = gamma + 1.;                                      //
        gpogm = gamp1 / gamm1;                                   //
        xg2   = xgeom + 2. - omega;                              //

        double denom2 = (2. * gamm1) + xgeom - (gamma * omega);  //
        double denom3 = (xgeom * (2. - gamma)) - omega;          //

        // Kamm equation 18 and 19
        double v2     = 4. / (   xg2 * gamp1);                   // Post-shock location
        double vstar  = 2. / ((gamm1 * xgeom) + 2.);             // Location of singular point vstar

        // Initialize output variables (Sedov solution)
        for(size_t i = 0; i < rPoints; i++)
        {
            rho[i] = 0.0;
            vel[i]  = 0.0;
            p[i]   = 0.0;
            u[i]   = 0.0;
            cs[i]  = 0.0;
        }

        // Set two logicals that determines the type of solution
        lsingular = false;
        lstandard = false;
        lvacuum   = false;
        if      (abs(v2 - vstar) <=         osmall) lsingular = true;
        else if (    v2          <  vstar - osmall) lstandard = true;
        else if (    v2          >  vstar + osmall) lvacuum   = true;

        // Set two apparent singularies: book's notation for omega2 and omega3
        lomega2 = false;
        lomega3 = false;
        if (abs(denom2) <= osmall) {
            lomega2 = true;
            denom2  = 1.e-8;
        } else if (abs(denom3) <= osmall) {
            lomega3 = true;
            denom3  = 1.e-8;
        }

        //  Various exponents in Kamm equations 42-47. In terms of book's notation:
        a0    =  2. / xg2;                                                                                                    // a0 =  beta6;
        a2    = -gamm1 / denom2;                                                                                              // a1 =  beta1;
        a1    =  xg2 * gamma / (2. + (xgeom * gamm1)) * (((2. * (xgeom * (2. - gamma) - omega)) / (gamma * xg2 * xg2)) - a2); // a2 = -beta2;
        a3    = (xgeom - omega) / denom2;                                                                                     // a3 =  beta3;
        a4    = xg2 * (xgeom - omega) * a1 / denom3;                                                                          // a4 =  beta4;
        a5    = ((omega * gamp1) - (2. * xgeom)) / denom3;                                                                    // a5 = -beta5;

        // Frequent combinations in Kamm equations 33-37
        a_val = 0.25 * xg2 * gamp1;                                                                                           //
        b_val = gpogm;                                                                                                        //
        c_val = 0.5 * xg2 * gamma;                                                                                            //
        d_val = (xg2 * gamp1) / ( (xg2 * gamp1) - (2. * (2. + (xgeom * gamm1))) );                                            //
        e_val = 0.5 * (2. + (xgeom * gamm1));                                                                                 //

        //  Evaluate the energy integrals.
        double alpha, vmin;

        if (lsingular)
        {
            // The singular case can be done by hand. It save some cpu cycles: Kamm equations 80, 81, and 85
            eval2 = gamp1 / (xgeom * pow((gamm1 * xgeom) + 2., 2.) );
            eval1 = 2. / gamm1 * eval2;

            alpha = (gpogm * pow(2.,xgeom)) / (xgeom * pow((gamm1 * xgeom) + 2., 2.));

            if (int(xgeom) != 1) alpha = M_PI * alpha;

        } else {

            // Set the radius coresponding to vv to zero for now
            rvv = 0.;

            //  Kamm equations 18, and 28.
            v0  = 2. / (xg2 * gamma);
            vv  = 2. /  xg2;

            if (lstandard)
            {
                // Post-shock origin
                vmin  = v0;

                // The first energy integral. In the standard case the term ((c_val * v) - 1.)) might be singular at 'v = vmin'
                gam_int = a3 - (a2 * xg2) - 1.;
                if (gam_int >= 0.){
                    qromo(efun01, vmin, v2, eps, eval1, midpnt);
                } else {
                    gam_int = abs(gam_int);
                    qromo(efun01, vmin, v2, eps, eval1, midpowl);
                }

                // The second energy integral. In the standard case the term ((c_val * v) - 1.) might be singular at 'v = vmin'
                gam_int = a3 - (a2 * xg2) - 2.;
                if (gam_int >= 0.) {
                    qromo(efun02, vmin, v2, eps, eval2, midpnt);
                } else {
                    gam_int = abs(gam_int);
                    qromo(efun02, vmin, v2, eps, eval2, midpowl);
                }
            }
            else if (lvacuum)
            {
                // Vacuum boundary
                vmin  = vv;

                // In the vacuum case the term (1 - (c_val / gamma * v)) might be singular at 'v = vv'
                gam_int = a5;
                if (gam_int >= 0.){
                    qromo(efun01, vmin, v2, eps, eval1, midpnt);
                } else {
                    gam_int = abs(gam_int);
                    qromo(efun01, vmin, v2, eps, eval1, midpowl2);
                }

                // In the vacuum case the term (1 - (c_val / gamma * v)) might be singular at 'v = vv'
                gam_int = a5;
                if (gam_int >= 0.){
                    qromo(efun02, vmin, v2, eps, eval2, midpnt);
                } else {
                    gam_int = abs(gam_int);
                    qromo(efun02, vmin, v2, eps, eval2, midpowl2);
                }
            }

            // In the Kamm equations 57 and 58 for alpha, in a slightly different form.
            if (dim == 1){
                alpha = (0.5 * eval1) + (eval2 / gamm1);
            } else {
                alpha = (xgeom - 1.) * M_PI * (eval1 + (2. * eval2 / gamm1));
            }
        }

        // Immediate post-shock values: Kamm page 14, equations 14, 16, 5, 13
        r2          = pow(eblast / (alpha * rho0), 1. / xg2) * pow(time, 2. / xg2);  // shock position
        double us   = (2. / xg2) * r2 / time;                                        // shock speed
        double rho1 = rho0 * pow(r2, -omega);                                        // pre-shock density
        double u2   = 2. * us / gamp1;                                               // post-shock material speed
        double rho2 = gpogm * rho1;                                                  // post-shock density
        double p2   = 2. * rho1 * pow(us, 2.) / gamp1;                               // post-shock pressure
        double e2   = p2 / (gamm1 * rho2);                                           // post-shoock specific internal energy
        double cs2  = sqrt(gamma * p2 / rho2);                                       // post-shock sound speed

        cout << endl;
        cout << "r2    = " << setw(39) << setprecision(38) << r2   << endl;
        cout << "us    = " << setw(39) << setprecision(38) << us   << endl;
        cout << "rho1  = " << setw(39) << setprecision(38) << rho1 << endl;
        cout << "u2    = " << setw(39) << setprecision(38) << us   << endl;
        cout << "rho2  = " << setw(39) << setprecision(38) << rho2 << endl;
        cout << "p2    = " << setw(39) << setprecision(38) << p2   << endl;
        cout << "e2    = " << setw(39) << setprecision(38) << e2   << endl;
        cout << "cs2   = " << setw(39) << setprecision(38) << cs2  << endl;
        cout << endl;

        // Find the radius corresponding to vv
        if (lvacuum){
            vwant = vv;
            rvv = zeroin(0., r2, sed_r_find, eps2);
            cout << endl << "I am in!" << endl;
        }

        // Loop over spatial positions
        for(size_t i = 0; i < rPoints; i++)
        {
            rwant = r[i];

            if (rwant > r2)
            {
                // If we are upstream from the shock front
                rho[i] = rho0 * pow(rwant, -omega);
                vel[i]  = vel0;
                p[i]   = p0;
                u[i]   = u0;
                cs[i]  = cs0;

                //cout << "Arrived_front! : " << i + 1 << ", rPoints = " << rPoints << endl;
            }
            else
            {
                // If we are between the origin and the shock front find the correct similarity value for this radius in the standard or vacuum cases
                double vat;

                //cout << endl;
                //cout << "v2   = " << setw(39) << setprecision(38) << v2   << endl;
                //cout << "v0   = " << setw(39) << setprecision(38) << v0   << endl;
                //cout << "vv   = " << setw(39) << setprecision(38) << vv   << endl;
                //cout << "eps2 = " << setw(39) << setprecision(38) << eps2 << endl;

                if      (lstandard) vat = zeroin(0.9 * v0,       v2, sed_v_find, eps2);
                else if (lvacuum)   vat = zeroin(      v2, 1.2 * vv, sed_v_find, eps2);
                else{
                    cout << "Error: lsingular case not expected" << endl;
                    exit(-1);
                }

                //cout << "Arrived_back! : " << i + 1 << ", rPoints = " << rPoints << endl;

                cout << "vat[" << i + 1 << "] = " << setw(39) << setprecision(38) << vat << endl;

                // The physical solution
                double l_fun, dlamdv, f_fun, g_fun, h_fun;

                sedov_funcs(vat, l_fun, dlamdv, f_fun, g_fun, h_fun);

                /*
                if (i == 0)
                {
                    cout << "i      = " << setw(39) << setprecision(38) << std::scientific << i + 1  << endl;
                    cout << "vat    = " << setw(39) << setprecision(38) << std::scientific << vat    << endl;
                    cout << "l_fun  = " << setw(39) << setprecision(38) << std::scientific << l_fun  << endl;
                    cout << "dlamdv = " << setw(39) << setprecision(38) << std::scientific << dlamdv << endl;
                    cout << "f_fun  = " << setw(39) << setprecision(38) << std::scientific << f_fun  << endl;
                    cout << "g_fun  = " << setw(39) << setprecision(38) << std::scientific << g_fun  << endl;
                    cout << "h_fun  = " << setw(39) << setprecision(38) << std::scientific << h_fun  << endl;
                }*/

                rho[i] = rho2 * g_fun;
                vel[i] = u2   * f_fun;
                p[i]   = p2   * h_fun;
                u[i]   = 0.;
                cs[i]  = 0.;

                if (rho[i] != 0.) {
                    u[i]  = p[i] / (gamm1 * rho[i]);
                    cs[i] = sqrt(gamma * p[i] / rho[i]);
                }


                //if (i == 0)
                //{
                    cout << "rho[" << i + 1 << "] = " << setw(39) << setprecision(38) << std::scientific << rho[i] << endl;
                    cout << "vel[" << i + 1 << "] = " << setw(39) << setprecision(38) << std::scientific << vel[i] << endl;
                    cout << "p  [" << i + 1 << "] = " << setw(39) << setprecision(38) << std::fixed      << p[i]   << endl;
                    cout << "u  [" << i + 1 << "] = " << setw(39) << setprecision(38) << std::fixed      << u[i]   << endl;
                    cout << "cs [" << i + 1 << "] = " << setw(39) << setprecision(38) << std::fixed      << cs[i]  << endl;
                    cout << endl;
                //}
            }

            /*
            if (i == 0)
            {
                cout << endl;
                cout << "rho[" << i + 1 << "] = " << setw(39) << setprecision(38) << std::scientific << rho[i] << endl;
                cout << "vel[" << i + 1 << "] = " << setw(39) << setprecision(38) << std::scientific << vel[i] << endl;
                cout << "p  [" << i + 1 << "] = " << setw(39) << setprecision(38) << std::fixed      << p[i]   << endl;
                cout << "u  [" << i + 1 << "] = " << setw(39) << setprecision(38) << std::fixed      << u[i]   << endl;
                cout << "cs [" << i + 1 << "] = " << setw(39) << setprecision(38) << std::fixed      << cs[i]  << endl;
            }*/
        }
    }

    static void sedov_funcs(const double v,           // Similarity variable v
                            double &     l_fun,       // out: l_fun is book's zeta
                            double &     dlamdv,      // out: l_fun derivative
                            double &     f_fun,       // out: f_fun is book's V
                            double &     g_fun,       // out: g_fun is book's D
                            double &     h_fun)       // out: h_fun is book's P
    {
        // Given the similarity variable v, returns functions: ' l_func: lambda', 'lambda_derivative', 'f', 'g', and 'h'
        // Although the ordinary differential equations are analytic, the sedov expressions appear to become singular for various combinations of parameters and at the lower limits of the integration range.
        // All these singularities are removable and done so by this routine.

        // Frequent combinations and their derivative with v. Kamm equation 29-32

        double x1    =  a_val * v;                          // x1 is book's F
        double dx1dv =  a_val;                              //

        double cbag  =  max(eps2, (c_val * v) - 1.);        //
        double x2    =  b_val * cbag;                       //
        double dx2dv =  b_val * c_val;                      //

        double ebag  =  1. - (e_val * v);                   //
        double x3    =  d_val * ebag;                       //
        double dx3dv = -d_val * e_val;                      //

        double x4    =  b_val * (1. - (0.5 * xg2 * v));     // x4 a bit different to save a divide
        double dx4dv = -b_val * 0.5 * xg2;                  //

        //  Transition region between standard and vacuum cases. Kamm page 15 or equations 88-92
        if (lsingular)
        {
            l_fun  = rwant / r2;
            dlamdv = 0.;
            f_fun  = l_fun;
            g_fun  = pow(l_fun, xgeom - 2.);
            h_fun  = pow(l_fun, xgeom);
        }
        else if (lvacuum && (rwant < rvv))
        {
            // For the vacuum case in the hole

            l_fun  = 0.;
            dlamdv = 0.;
            f_fun  = 0.;
            g_fun  = 0.;
            h_fun  = 0.;
        }
        else if (lomega2)
        {
            // omega = omega2 = (2*(gamma -1) + xgeom)/gamma case, denom2 = 0. Book expressions 20-22

            double beta0  =  1. / (2. * e_val);
            double c6     =  0.5 * gamp1;
            double c2     =  c6 / gamma;
            double y      =  1. / (x1 - c2);
            double z      = (1. - x1) * y;
            double dpp2dv = -gamp1 * beta0 * dx1dv * y * (1. + z);

            double pp1    =  gamm1 * beta0;
            double pp2    =  gamp1 * beta0 * z;
            double pp3    = (4. - xgeom - (2. * gamma)) * beta0;
            double pp4    = -xgeom * gamma * beta0;

            l_fun  = pow(x1, -a0) * pow(x2, pp1) * exp(pp2);
            dlamdv = ( (-a0 * dx1dv / x1) + (pp1 * dx2dv / x2) + dpp2dv) * l_fun;
            f_fun  = x1 * l_fun;
            g_fun  = pow(x1, a0 * omega) * pow(x2, pp3) * pow(x4,      a5) * exp(-2. * pp2);
            h_fun  = pow(x1, a0 * xgeom) * pow(x2, pp4) * pow(x4, 1. + a5);
        }
        else if (lomega3)
        {
            // omega = omega3 = xgeom*(2 - gamma) case, denom3 = 0. Book expressions 23-25

            double beta0  = 1. / (2. * e_val);
            double c6     = 0.5 * gamp1;

            double pp1    = a3 + omega * a2;
            double pp2    = 1. - 4. * beta0;
            double pp3    = -xgeom * gamma * gamp1 * beta0 * (1. - x1) / (c6 - x1);
            double pp4    = 2. * (xgeom * gamm1 - gamma) * beta0;

            l_fun  = pow(x1, -a0) * pow(x2, -a2) * pow(x4, -a1);
            dlamdv = -( (a0 * dx1dv / x1) + (a2 * dx2dv / x2) + (a1 * dx4dv / x4) ) * l_fun;
            f_fun  = x1 * l_fun;
            g_fun  = pow(x1, a0 * omega) * pow(x2, pp1) * pow(x4, pp2) * exp(pp3);
            h_fun  = pow(x1, a0 * xgeom)                * pow(x4, pp4) * exp(pp3);
        }
        else
        {
            // For the standard or vacuum case not in the hole. Kamm equations 38-41

            l_fun  = pow(x1, -a0) * pow(x2, -a2) * pow(x3, -a1);
            dlamdv = -( (a0 * dx1dv / x1) + (a2 * dx2dv / x2) + (a1 * dx3dv / x3) ) * l_fun;
            f_fun  = x1 * l_fun;
            g_fun  = pow(x1, a0 * omega) * pow(x2, a3 + (a2 * omega)) * pow(x3, a4 + (a1 *  omega      )) * pow(x4,      a5);
            h_fun  = pow(x1, a0 * xgeom)                              * pow(x3, a4 + (a1 * (omega - 2.))) * pow(x4, 1. + a5);
        }
    }

    static double efun01(const double v)
    {
        // Evaluates the first energy integrand, kamm equations 67 and 10.
        // The (c_val*v - 1) term might be singular at v=vmin in the standard case.
        // The (1 - c_val/gamma * v) term might be singular at v=vmin in the vacuum case.
        // Due care should be taken for these removable singularities by the integrator.

        double l_fun, dlamdv, f_fun, g_fun, h_fun;

        sedov_funcs(v, l_fun, dlamdv, f_fun, g_fun, h_fun);

        return (dlamdv * pow(l_fun, xgeom + 1.) * gpogm * g_fun * pow(v, 2.));
    }

    static double efun02(const double v)
    {
        // Evaluates the second energy integrand, kamm equations 68 and 11.
        // The (c_val*v - 1) term might be singular at v=vmin in the standard case.
        // The (1 - c_val/gamma * v) term might be singular at v=vmin in the vacuum case.
        // Due care should be taken for these removable singularities by the integrator.

        double l_fun, dlamdv, f_fun, g_fun, h_fun;

        sedov_funcs(v, l_fun, dlamdv, f_fun, g_fun, h_fun);

        double z = 8. / ( pow(xgeom + 2. - omega, 2.) * gamp1);

        return (dlamdv * pow(l_fun, xgeom - 1.) * h_fun * z);
    }

    static double sed_v_find(const double v)
    {
        // Given corresponding physical distances, find the similarity variable v. Kamm equation 38 as a root find
        double l_fun, dlamdv, f_fun, g_fun, h_fun;

        sedov_funcs(v,l_fun,dlamdv,f_fun,g_fun,h_fun);

        return ((r2 * l_fun) - rwant);
    }

    static double sed_r_find(const double r)
    {
        // Given the similarity variable v, find the corresponding physical distance. Kamm equation 38 as a root find
        double l_fun, dlamdv, f_fun, g_fun, h_fun;

        sedov_funcs(vwant,l_fun,dlamdv,f_fun,g_fun,h_fun);

        return ((r2 * l_fun) - r);
    }

    static void midpnt(const size_t                   n,      //
                       function<double(const double)> func,   //
                       const double                   a,      //
                       const double                   b,      //
                       double &                       s)      //
    {
        // This routine computes the n'th stage of refinement of an extended midpoint rule.
        // Func is input as the name of the function to be integrated between limits a and b.
        // When called with n=1, the routine returns as s the crudest estimate of the integralof func from a to b.
        // Subsequent calls with n=2,3... improve the accuracy of s by adding 2/3*3**(n-1) addtional interior points.

        if (n == 1)
        {
            double x = 0.5 * (a + b);
            s = (b - a) * func(x);
        }
        else
        {
            size_t i_max = pow(3, n - 2);
            double tnm   = double(i_max);

            double del   = (b - a) / (3. * tnm);
            double ddel  = 2. * del;

            double sum   = 0.;
            double x     = a + (0.5 * del);

            for (size_t i = 0; i < i_max; i++)
            {
                sum = sum + func(x);
                x   =   x + ddel;

                sum = sum + func(x);
                x   =   x + del;
            }

            s  = ( s + ( (b - a) * sum / tnm ) ) / 3.;
        }
    }

    static double midpowl_func(function<double(const double)> funk,   //
                               const double                   x,      //
                               const double                   aa)     //
    {
        // A little conversion, recipe equation 4.4.3

        double p1 =                       (1. - gam_int);
        double p2 =      pow(x, gam_int / (1. - gam_int));
        double p3 = funk(pow(x,      1. / (1. - gam_int)) + aa);

        return (1. / p1 * p2 * p3);
    }

    static void midpowl(const size_t                   n,             //
                        function<double(const double)> funk,          //
                        const double                   aa,            //
                        const double                   bb,            //
                        double &                       s)             //
    {
        // This routine is an exact replacement for midpnt,
        // except that it allows for an integrable power-law singularity
        // of the form pow(x - a, -gam_int)
        // at the lower limit aa for 0 < gam_int < 1.

        double b = pow(bb - aa, 1. - gam_int);
        double a = 0.;

        // Now exactly as midpnt
        if (n == 1)
        {
            double x = 0.5 * (a + b);

            s = (b - a) * midpowl_func(funk, x, aa);
        }
        else
        {
            size_t i_max = pow(3, n - 2);
            double tnm   = double(i_max);

            double del   = (b - a) / (3. * tnm);
            double ddel  = 2. * del;

            double sum   = 0.;
            double x     = a + (0.5 * del);

            for (size_t i = 0; i < i_max; i++)
            {
                sum = sum + midpowl_func(funk, x, aa);
                x   =   x + ddel;

                sum = sum + midpowl_func(funk, x, aa);
                x   =   x + del;
            }

            s = ( s + ( (b - a) * sum / tnm ) ) / 3.;
        }
    }

    static double midpowl2_func(function<double(const double)> funk,  //
                              const double                     x,     //
                              const double                     aa)    //
    {
        // A little conversion, module recipe equation 4.4.3

        double p1 =                            (gam_int - 1.);
        double p2 =           pow(x, gam_int / (1. - gam_int));
        double p3 = funk(aa - pow(x,      1. / (1. - gam_int)));

        return (1. / p1 * p2 * p3);
    }

    static void midpowl2(const size_t                   n,            //
                         function<double(const double)> funk,         //
                         const double                   aa,           //
                         const double                   bb,           //
                         double &                       s)            //
    {
        // This routine is an exact replacement for midpnt,
        // except that it allows for an integrable power-law singularity
        // of the form pow(a - x, -gam_int)
        // at the lower limit aa for 0 < gam_int < 1.

        double b = pow(aa - bb, 1. - gam_int);
        double a = 0.;

        // Now exactly as midpnt
        if (n == 1)
        {
            double x = 0.5 * (a + b);

            s = (b - a) * midpowl2_func(funk, x, aa);
        }
        else
        {
            size_t i_max = pow(3, n - 2);
            double tnm   = double(i_max);

            double del   = (b - a) / (3. * tnm);
            double ddel  = 2. * del;

            double sum   = 0.;
            double x     = a + (0.5 * del);

            for (size_t i = 0; i < i_max; i++)
            {
                sum = sum + midpowl2_func(funk, x, aa);
                x   =   x + ddel;

                sum = sum + midpowl2_func(funk, x, aa);
                x   =   x + del;
            }

            s = (s + ((b - a) * sum / tnm)) / 3.;
        }
    }

    static void polint(double *     xa,     //
                       double *     ya,     //
                       const size_t n,      //
                       const double x,      //
                       double &     y,      // out:
                       double &     dy)     // out:
    {
        // Given arrays xa and ya of length n and a value x, this routine returns a value y and an error estimate dy.
        // if p(x) is the polynomial of degree n-1 such that ya = p(xa) ya then the returned value is y = p(x)

        const size_t nmax = 20;

        double c[nmax];
        double d[nmax];

        // Find the index ns of the closest table entry; initialize the c and d tables
        size_t ns  = 0;
        double dif = abs(x - xa[0]);

        for(size_t i = 0; i < n; i++)
        {
            double dift = abs(x - xa[i]);

            if (dift < dif)
            {
                ns  = i;
                dif = dift;
            }

            c[i]  = ya[i];
            d[i]  = ya[i];
        }

        // First guess for y
        y = ya[ns];

        // for each column of the table, loop over the c's and d's and update them
        ns = ns - 1;

        for (size_t m = 0; m < n - 1; m++)
        {
            for (size_t i = 0; i < n - m - 1; i++)
            {
                double ho = xa[i        ] - x;
                double hp = xa[i + m + 1] - x;

                double w  =  c[i + 1] - d[i];

                double den  = ho - hp;

                if (den == 0.){
                    cout << "Error: Two 'xa' positions are the same in polint" << endl;
                    exit(-1);
                }

                den  =  w / den;

                d[i] = hp * den;
                c[i] = ho * den;
            }

            // After each column is completed, decide which correction c or d, to add to the accumulating value of y,
            // that is, which path to take in the table by forking up or down.
            if (2 * (ns + 1) < n - m - 1)
            {
                dy = c[ns + 1];
            }
            else
            {
                dy = d[ns];

                // ns is updated as we go to keep track of where we are.
                ns = ns - 1;
            }

            // The last dy added is the error indicator.
            y = y + dy;
        }
    }

    static void qromo(function<double(const double)> func,
                      const double a,
                      const double b,
                      const double eps,
                      double &     ss,
                      function<void(const size_t, function<double(const double)>, const double, const double, double &)> choose)
    {
        // This routine returns as 's' the integral of the function 'func'
        // from 'a' to 'b' with fractional accuracy 'eps'.
        // Integration is done via Romberg algorithm.

        // It is assumed the call to 'choose' triples the number of steps on each call
        // and that its error series contains only even powers of the number of steps.
        // The external choose may be any of the above drivers, i.e midpnt,midinf...

        const size_t imax  = 14;
        const size_t imaxp = imax + 1;

        const size_t j     = 5;
        const size_t jm    = j - 1;

        double s[imaxp];
        double h[imaxp];

        h[0] = 1.;

        // jmax limits the number of steps; nsteps = 3**(jmax-1)
        for (size_t i = 0; i < imax; i++)
        {
            // Integrate function
            choose(i + 1,                // Num of ref steps         'n'
                   func,                 // function to be integrate 'func'
                   a,                    // bottom limit             'a'
                   b,                    // top    limit             'b'
                   s[i]);                // Out: integration value   's'

            if (i >= j - 1)
            {
                double dss;

                polint( &h[i - jm],  // array pointer 'xa'
                        &s[i - jm],  // array pointer 'ya'
                        j,               // size          'n'
                        0.,              // value         'x'
                        ss,              // Out: value    'y'
                        dss);            // Out: error    'dy'

                if (abs(dss) <= eps * abs(ss)) return;
            }

            s[i + 1] = s[i];
            h[i + 1] = h[i] / 9.;
        }

        cout << "Error: Too many steps in qromo" << endl;
    }

    static double zeroin(const double                   ax,   // Left endpoint of initial interval
                         const double                   bx,   // Right endpoint of initial interval
                         function<double(const double)> f,    // Function subprogram which evaluates f(x) for any x in the interval [ax,bx]
                         const double                   tol)  // Desired length of the interval of uncertainty of the final result (>= 0.)
    {
        // This subroutine solves the 'zeroin' abcissa approximating to zero of 'f' in the interval [ax,bx]

        // It is assumed  that 'f(ax)' and 'f(bx)' have  opposite  signs without a check.

        // Returns a zero 'x' in the given interval [ax,bx]  to within a tolerance  '4*macheps*abs(x) + tol'
        // , where macheps is the relative machine precision.

        // This function subprogram is a slightly  modified  translation of the algol 60 procedure zero given in Richard Brent
        // , algorithms for minimization without derivatives, prentice - hall, inc. (1973).

        //cout << endl;
        //cout << "ax  = " << ax  << endl;
        //cout << "bx  = " << bx  << endl;
        //cout << "tol = " << tol << endl;

        // Compute 'eps', the relative machine precision
        double eps = 1.;
        double tol1;
        do {
          eps  = eps / 2.;
          tol1 = 1. + eps;
        } while (tol1 > 1.);

        // Initialization : Begin step

        double a  = ax;
        double b  = bx;
        double c  = a;

        double fa = f(a);
        double fb = f(b);
        double fc = fa;

        double d  = b - a;
        double e  = d;

        while(true)
        {
            // Update variables for the current try
            if (abs(fc) < abs(fb))
            {
                a  = b;
                b  = c;
                c  = a;
                fa = fb;
                fb = fc;
                fc = fa;
            }

            // Convergence test
            tol1 = (2. * eps * abs(b)) + (0.5 * tol);

            // Update xm
            double xm = 0.5 * (c - b);

            // Is the solution found?
            if ( (abs(xm) <= tol1) || (fb == 0.)  )
            {
                // Yes! Return solution
                return b;
            }

            // Is bisection necessary?
            if ( (abs(e) < tol1) || (abs(fa) <= abs(fb)) )
            {
                // Update e,d
                d = xm;
                e = d;
            }
            else
            {
                // Local variables
                double p,q,r,s;

                // Is quadratic interpolation possible?
                if (a != c)
                {
                    // Use inverse quadratic interpolation
                    q = fa / fc;
                    r = fb / fc;
                    s = fb / fa;

                    p = s * ( (2. * xm * q * (q - r)) - ((b - a) * (r - 1.)) );
                    q = (q - 1.) * (r - 1.) * (s - 1.);

                } else {

                    // Use linear interpolation
                    s = fb / fa;
                    p = 2. * xm * s;
                    q = 1. - s;
                }

                // Adjust signs
                if (p > 0.) q = -q;
                p = abs(p);

                // Is interpolation acceptable?
                if (    ( (2. * p) >= ((3. * xm * q) - abs(tol1 * q)) )
                     || (       p  >= abs(0.5 * e * q)                ) )
                {
                    // Bisection
                    d = xm;
                    e = d;
                }
                else
                {
                    // Update e,d
                    e = d;
                    d = p / q;
                }
            }

            // Complete step
            a  = b;
            fa = fb;

            // Update b
            if (abs(d) > tol1) b += d;
            else {
                if (xm >= 0.)  b += abs(tol1);
                else           b -= abs(tol1);
            }

            // Update fb
            fb = f(b);

            // Update variables for the next step
            if ((fb * (fc / abs(fc))) > 0.)
            {
                c  = a;
                fc = fa;
                d  = b - a;
                e  = d;
            }
        }
    }
};

double SedovAnalyticalSolution::xgeom;
double SedovAnalyticalSolution::omega;
double SedovAnalyticalSolution::gamma;

double SedovAnalyticalSolution::gamm1;
double SedovAnalyticalSolution::gamp1;
double SedovAnalyticalSolution::gpogm;
double SedovAnalyticalSolution::xg2;

bool   SedovAnalyticalSolution::lsingular;
bool   SedovAnalyticalSolution::lstandard;
bool   SedovAnalyticalSolution::lvacuum;

bool   SedovAnalyticalSolution::lomega2;
bool   SedovAnalyticalSolution::lomega3;

double SedovAnalyticalSolution::a0;
double SedovAnalyticalSolution::a1;
double SedovAnalyticalSolution::a2;
double SedovAnalyticalSolution::a3;
double SedovAnalyticalSolution::a4;
double SedovAnalyticalSolution::a5;

double SedovAnalyticalSolution::a_val;
double SedovAnalyticalSolution::b_val;
double SedovAnalyticalSolution::c_val;
double SedovAnalyticalSolution::d_val;
double SedovAnalyticalSolution::e_val;

double SedovAnalyticalSolution::rwant;
double SedovAnalyticalSolution::vwant;

double SedovAnalyticalSolution::r2;
double SedovAnalyticalSolution::v0;
double SedovAnalyticalSolution::vv;
double SedovAnalyticalSolution::rvv;

double SedovAnalyticalSolution::gam_int;

