// © 2024 Hiro Wakimura, Feng Xiao
// MUSCL-THINC-BVD scheme
// for compressible two-phase flow
//------------------------------------------------------------

#include<iostream>
#include<fstream>
#include<cstdio>
#include<cstdlib>
#include<cstring>
#define _USE_MATH_DEFINES
#include<cmath>
#include<algorithm>
#include<vector>
#include<valarray>
#include<iomanip>
#include<omp.h>
#include <unordered_map>

#define eps 1.0e-15
#define GNUPLOT "gnuplot -persist"
#define SYMP 1

using std::cin;
using std::cout;
using std::isnan;
using std::vector;
using std::endl;
using std::string;
using std::min;
using std::max;

typedef std::vector<int> vec1i;
typedef std::vector<vec1i> vec2i;
typedef std::vector<vec2i> vec3i;
typedef std::vector<vec3i> vec4i;
typedef std::vector<vec4i> vec5i;
typedef std::vector<double> vec1d;
typedef std::vector<vec1d> vec2d;
typedef std::vector<vec2d> vec3d;
typedef std::vector<vec3d> vec4d;
typedef std::vector<vec4d> vec5d;

vec1i loop_begin_base, loop_size_base;

// global variables
int dim; // number of spatial dimension (1, 2, or 3)
int EOS_type; // 1: ideal gas, 2: stiffened gas, 3: Mie-Gruneisen EOS
int problem_type; // 1: 2D static droplet problem
int scheme_type; // 1: MUSCL, 2: THINC, 3: MUSCL-THINC-BVD
string scheme_name; // name of numerical scheme
int num_var; // number of variables in 5eq model (alpha1, alpha1rho1, alpha2rho2, rhou, rhov, rhow, rhoE)
int nx, ngx, NX, NBX, ny, ngy, NY, NBY, nz, ngz, NZ, NBZ, NN; // number of cell in x, y, z-direction in computational domain, including ghost cells
int ns; // number of cells in stencil for reconstruction
double x_range[2], dx, y_range[2], dy, z_range[2], dz; // range of computational domain in x, y, z-direction
vec1d xc, xb, yc, yb, zc, zb; // x, y, z-coordinate at cell center and cell boundary
double t, t_end, dt, dt_last; // time, end time, time step, last time step
int t_step; // time step number
vec1i boundary_type; // boundary condition type (1: outflow)
std::string material; // material name for equation of state
vec3d U; // conservative variables (alpha1, alpha1rho1, alpha2rho2, rhou, rhov, rhow, rhoE)
double beta, T1_THINC; // gradient parameter and precalculated value for THINC scheme
int BVD, BVD_s; // number of candidate reconstruction schemes, number of cells to be expanded outside the computational domain for the BVD scheme
vec4i BVD_active; // record cells where MUSCL has been replaced by THINC
vec3d W_x_L, W_x_R, W_y_L, W_y_R, W_z_L, W_z_R; // reconstructed left-side and right-side cell boundary value in x, y, z-direction
int bvc_check; // flag for BVC (boundary value correction)
int spatial_discre; // spatial discretization method (1: divergence of flux tensor, 2: wave-propagation method)
vec2d F_x, F_y, F_z; // flux vector in x, y, z-direction
vec2d Apdq_x, Amdq_x, Adq_x, Apdq_y, Amdq_y, Adq_y, Apdq_z, Amdq_z, Adq_z; // fluctuation in x, y, z-direction for wave-propagation method
double CFL; // Courant number
int RK_type; // type of Runge Kutta method (1: 1-stage, 2:2-stage 2nd-order, 3: 3-stage 3rd-order)
int RK_stage; // stage number in Runge Kutta method
vec2d RK_alpha; // coefficients of Runge Kutta method
vec1d RK_beta; // coefficients of Runge Kutta method
int num_threads; // number of threads for OpenMP

double gamma_,gamma1,gamma2,pi1,pi2,eta1,eta2,eta1d,eta2d,Cv1,Cv2,Cp1,Cp2,As,Bs,Cs,Ds;
double cB11,cB12,cB21,cB22,cE11,cE12,cE21,cE22,rho01,rho02,e01,e02;
int sound_speed_type; // 1: system sound speed, 2: mixture-mixture sound speed

int surface_tension_type; // 1: linear polynomial, 2: linear + filtering
int order_grad_VOF; // order of gradient of VOF for surface tension calculation
int ngx_CSF, ngy_CSF, ngz_CSF, NX_CSF, NY_CSF, NZ_CSF, NN_CSF; // number of ghost cells, total number of cells in x, y, z-direction for CSF
double sigma_CSF;
vec2d grad_VOF,normal_vec;
vec1d curv;
vec1d VOF_x_Riemann, VOF_y_Riemann, VOF_z_Riemann; // VOF value at cell boundary in x, y, z-direction derived from Riemann solver
vec1d u_Riemann, v_Riemann, w_Riemann; // velocity at cell boundary in x, y, z-direction derived from Riemann solver
vec1d weight_filter, weight_sum_filter, curv_sum_filter;
vec2d F_CSF;

vec1d coef_Pn_D1_CC;

void parameter();
std::unordered_map<string, string> read_params(const std::string&);
void EOS_param();
void EOS_param_ref
(
    double *Gammak, double *p_refk, double *e_refk, 
    double rhok, double gammak, double pik, double etak,
    double cB1k, double cB2k, double cE1k, double cE2k, double rho0k, double e0k
);
double sound_speed_square
(
    double rhok, double pk,
    double gammak, double pik, double etak,
    double cB1k, double cB2k, double cE1k, double cE2k, double rho0k, double e0k
);
void initial_condition();
void initial_condition_1D_GasLiquidRiemann();
void initial_condition_2D_static_droplet();
void prim_to_cons_5eq(double *alpha1rho1,double *alpha2rho2,double *rhou,double *rhov,double *rhow,double *rhoE,
                  double alpha1,double rho1,double rho2,double u,double v,double w,double p);
void cons_to_prim_5eq(double *rho1,double *rho2,double *u,double *v,double *w,double *p,
                  double alpha1,double alpha1rho1,double alpha2rho2,double rhou,double rhov,double rhow,double rhoE);
void initialize_BVD_active();
void boundary_condition(int);
void reconstruction_dim(int d, int rk, vec3d& W_L, vec3d& W_R);
void MUSCL(double*, double*, const vec1d&);
double Phi_MUSCL(double);
void THINC(double*, double*, const vec1d&);
void BVD_selection_dim(int d, int rk, vec3d& W_L, vec3d& W_R, vec3i&);
// void bvc_Euler_x(int);
void Riemann_solver_5eq_HLLC_dim(int d, double *MWS_d, const vec3d& W_L, const vec3d& W_R, vec2d& F, vec2d& Amdq, vec2d& Apdq, vec2d& Adq, vec1d& vel_Riemann, vec1d& col_Riemann);
void cal_dt(double,double,double);
void update(int);
void output_result(string);
double sign(double);
inline int I_c(int, int, int);
inline int I_x(int, int, int);
inline int I_y(int, int, int);
inline int I_z(int, int, int);
inline void get_cb_indices_from_cc(int, int, int, int, int&, int&);
double q2(double,double,double);
double sum_cons3(double, double, double);
double pow_int(double x,int n);
void CSFmodel(int);
void grad_VOF_cal_dim(int, int);
void curv_cal();
void curv_filter(int);
void coefficient_linear_polynoimal_1stDerivative_CellCenter(int n);

int main(void){
    
    int i, d, m, rk, per;
    double percent, MWS_x, MWS_y, MWS_z;
    int file_num = 0;
    string file_name;
    
    parameter();
    
    initial_condition();
    
    // output initial condition
    file_name = "./data/result0.csv";
    output_result(file_name);
    
    t = 0.; t_step = 0; per = 1; dt_last = 0.; dt = 0.0; bvc_check = 0; // initialization
    MWS_x = 0.0; MWS_y = 0.0; MWS_z = 0.0; // initialization of MWS (maximum wave speed)
    while (1){
        // adjust value of dt in final time step
        if (t + dt > t_end) dt_last = t_end - t;
        
        if (BVD > 1) initialize_BVD_active();
        
        for (rk = 0; rk < RK_stage; rk++){
            // set values in ghost cells
            boundary_condition(rk);
            
            // interpolate cell boundary values
            // reconstruction(rk);
            if (dim >= 1) reconstruction_dim(0, rk, W_x_L, W_x_R);
            if (dim >= 2) reconstruction_dim(1, rk, W_y_L, W_y_R);
            if (dim >= 3) reconstruction_dim(2, rk, W_z_L, W_z_R);
            
            // calculate carvature for surface tension
            if (surface_tension_type == 1 || surface_tension_type == 2) CSFmodel(rk);
            
            // calculate numerical flux
            // Riemann_solver_5eq_HLLC(&MWS_x, &MWS_y, &MWS_z);
            if (dim >= 1) Riemann_solver_5eq_HLLC_dim(0, &MWS_x, W_x_L, W_x_R, F_x, Amdq_x, Apdq_x, Adq_x, u_Riemann, VOF_x_Riemann);
            if (dim >= 2) Riemann_solver_5eq_HLLC_dim(1, &MWS_y, W_y_L, W_y_R, F_y, Amdq_y, Apdq_y, Adq_y, v_Riemann, VOF_y_Riemann);
            if (dim >= 3) Riemann_solver_5eq_HLLC_dim(2, &MWS_z, W_z_L, W_z_R, F_z, Amdq_z, Apdq_z, Adq_z, w_Riemann, VOF_z_Riemann);
            
            // update numerical solution
            cal_dt(MWS_x, MWS_y, MWS_z);
            update(rk);

        }
        t += dt;
        t_step++;
        
        // display of calculation progress
        percent = t / t_end * 100.0;
        if (percent >= per * 10 - eps){
            cout << per * 10 << "%  ";
            fflush(stdout);
            per++;
        }
        
        // output result
        int result_time_type = 2; // 1: output every xx time step, 2: output every xx% of calculation progress
        int result_interval = 1; // number of time steps for output
        int result_num = 10; // number of result files to be output
        // output result every time step
        if (result_time_type==1){
            if (t_step%result_interval==0){
                file_num++;
                file_name = "./data/result" + std::to_string(file_num) + ".csv";
                output_result(file_name);
                printf("t=%f,t_step=%d\n",t,t_step); getchar();
            }
        }
        // output result every 10% of calculation progress
        else if (result_time_type==2){
            if (percent/100.0>(double)(file_num+1)/result_num+eps){
                file_num++;
                file_name = "./data/result" + std::to_string(file_num) + ".csv";
                output_result(file_name);
            }
        }
        
        if (t >= t_end) break;
    }
    
    cout << "\nt_step = " << t_step << ", t = " << t << endl;
    if (bvc_check == 1) cout << "bvc is activated" << endl;
    
    output_result("./data/result.csv");

    return 0;
}

void parameter(){
    int i, j, k;
    
    auto params = read_params("param.txt");
    
    dim                  = std::stoi(params.at("dimension"));
    problem_type         = std::stoi(params.at("problem_type"));
    scheme_type          = std::stoi(params.at("scheme"));
    surface_tension_type = std::stoi(params.at("surface_tension"));
    RK_type              = std::stoi(params.at("RK_type"));
    num_threads          = std::stoi(params.at("num_threads"));
    
    string scheme_name;
    if      (scheme_type == 1) scheme_name = "MUSCL";
    else if (scheme_type == 2) scheme_name = "THINC";
    else if (scheme_type == 3) scheme_name = "MUSCL-THINC-BVD";
    else {
        std::cerr << "Invalid scheme value." << std::endl;
        exit(0);
    }
    if (scheme_name == "MUSCL" || scheme_name == "THINC") BVD = 1; // number of cacndidate reconstruction schemes
    else if (scheme_name == "MUSCL-THINC-BVD") BVD = 2;

    if (dim < 1 || dim > 3) {
        std::cerr << "Invalid dimension: must be 1, 2, or 3." << std::endl;
        exit(0);
    }
    
    // int num_threads = omp_get_max_threads(); // get maximum number of threads
    omp_set_num_threads(num_threads); // set number of threads for OpenMP

    // Print the loaded settings
    std::cout << "Dimension:        " << dim << "\n";
    std::cout << "Problem type:     " << problem_type << "\n";
    std::cout << "Scheme:           " << scheme_name << "\n";
    std::cout << "Surface tension:  " << surface_tension_type << "\n";
    std::cout << "RK type:          " << RK_type << "\n";
    
    num_var = 7; // number of variables in 5eq model (alpha1, rho1, rho2, rhou, rhov, rhow, rhoE)
    
    BVD_s = BVD - 1; // number of cells to be expanded outside the computational domain for the BVD scheme
    ns = 3; // number of cells in stencil for reconstruction
    ngx = 2 + BVD_s; // number of ghost cells for x-direction
    ngy = ngx; // number of ghost cells for y-direction
    ngz = ngx; // number of ghost cells for z-direction
    
    if (dim == 1){
        // 1D shock-tube problem
        if (problem_type == 1){
            nx = 200; // number of cell in x-direction in computational domain
            x_range[0] = -1.0; // x_coordinate at left boundary of computational domain
            x_range[1] = 1.0; // x_coordinate at right boundary of computational domain
            boundary_type = {1, 1}; // boundary condition type (1: outflow)
            material = "water-air2_nondim"; // material name for equation of state
            sound_speed_type = 2; // system sound speed
        }
        
        else if (problem_type == 2){
            
        }
        
        ny = 1; // number of cell in y-direction in computational domain
        ngy = 0; // number of ghost cells in y-direction
        nz = 1; // number of cell in z-direction in computational domain
        ngz = 0; // number of ghost cells in z-direction
        y_range[0] = -0.5; // y_coordinate at left boundary of computational domain
        y_range[1] = 0.5; // y_coordinate at right boundary of computational domain
        z_range[0] = -0.5; // z_coordinate at left boundary of computational domain
        z_range[1] = 0.5; // z_coordinate at right boundary of computational domain
    }
    else if (dim == 2){
        // 2D static droplet problem
        if (problem_type == 1){
            nx = 100/2; // number of cell in x-direction in computational domain
            ny = 100/2; // number of cell in y-direction in computational domain
            x_range[0] = -2.0; // x_coordinate at left boundary of computational domain
            x_range[1] = 2.0; // x_coordinate at right boundary of computational domain
            y_range[0] = -2.0; // y_coordinate at left boundary of computational domain
            y_range[1] = 2.0; // y_coordinate at right boundary of computational domain
            boundary_type={1, 1, 1, 1, -1, -1};
            material="water-air1_nondim"; // material name for equation of state
            sound_speed_type = 1; // system sound speed
        }
        
        else if (problem_type == 2){
            
        }
        
        nz = 1;   // number of cell in z-direction in computational domain
        ngz = 0; // number of ghost cells in z-direction
        z_range[0] = -0.5; // z_coordinate at left boundary of computational domain
        z_range[1] = 0.5; // z_coordinate at right boundary of computational domain
    }

    else if (dim == 3){
        
    }
    
    loop_begin_base = {ngx, ngy, ngz};
    loop_size_base = {nx, ny, nz};
    
    NX = nx + ngx * 2; // total number of cell in x-direction including ghost cell
    NY = ny + ngy * 2; // total number of cell in y-direction including ghost cell
    NZ = nz + ngz * 2; // total number of cell in z-direction including ghost cell
    NN = NX * NY * NZ; // total number of cell including ghost cell
    NBX = (NX + 1) * NY * NZ; // total number of cell boundary facing x-direction including ghost cell
    NBY = NX * (NY + 1) * NZ; // total number of cell boundary facing y-direction including ghost cell
    NBZ = NX * NY * (NZ + 1); // total number of cell boundary facing z-direction including ghost cell
    
    dx = (x_range[1] - x_range[0]) / nx; // cell length in x-direction
    dy = (y_range[1] - y_range[0]) / ny; // cell length in y-direction
    dz = (z_range[1] - z_range[0]) / nz; // cell length in z-direction
    xc = vec1d(NX); // x-coordinate at cell center
    yc = vec1d(NY); // y-coordinate at cell center
    zc = vec1d(NZ); // z-coordinate at cell center
    for (i = 0; i < NX; i++) xc[i] = x_range[0] + dx / 2.0 + dx * (i - ngx);
    for (j = 0; j < NY; j++) yc[j] = y_range[0] + dy / 2.0 + dy * (j - ngy);
    for (k = 0; k < NZ; k++) zc[k] = z_range[0] + dz / 2.0 + dz * (k - ngz);
    xb = vec1d(NX + 1); // x-coordinate at cell boundary
    yb = vec1d(NY + 1); // y-coordinate at cell boundary
    zb = vec1d(NZ + 1); // z-coordinate at cell boundary
    for (i = 0; i < NX + 1; i++) xb[i] = x_range[0] + dx * (i - ngx);
    for (j = 0; j < NY + 1; j++) yb[j] = y_range[0] + dy * (j - ngy);
    for (k = 0; k < NZ + 1; k++) zb[k] = z_range[0] + dz * (k - ngz);
    
    EOS_param(); // set parameters in equation of state
    
    if (RK_type ==1){ // coefficients of 1-stage Runge Kutta method
        RK_stage = 1;
        RK_alpha = vec2d(RK_stage, vec1d(RK_stage));
        RK_alpha[0][0] = 1.0;
        RK_beta = vec1d(RK_stage);
        RK_beta[0] = 1.0;
    }
    else if (RK_type == 2){ // coefficients of 2-stage 2nd-order Runge Kutta method
        RK_stage = 2;
        RK_alpha = vec2d(RK_stage, vec1d(RK_stage));
        RK_alpha[0][0] = 1.0; RK_alpha[0][1] = 0.0;
        RK_alpha[1][0] = 0.5; RK_alpha[1][1] = 0.5;
        RK_beta = vec1d(RK_stage);
        RK_beta[0] = 1.0; RK_beta[1] = 0.5;
    }
    else if (RK_type == 3){ // coefficients of 3-stage 3rd-order Runge Kutta method
        RK_stage = 3;
        RK_alpha = vec2d(RK_stage, vec1d(RK_stage));
        RK_alpha[0][0] = 1.0; RK_alpha[0][1] = 0.0; RK_alpha[0][2] = 0.0;
        RK_alpha[1][0] = 0.75; RK_alpha[1][1] = 0.25; RK_alpha[1][2] = 0.0;
        RK_alpha[2][0] = 1.0/3.0; RK_alpha[2][1] = 0.0; RK_alpha[2][2] = 2.0/3.0;
        RK_beta = vec1d(RK_stage);
        RK_beta[0] = 1.0; RK_beta[1] = 0.25; RK_beta[2] = 2.0/3.0;
        // RK_alpha = {{1.0, 0.0, 0.0},
        //             {0.75, 0.25, 0.0},
        //             {1.0/3.0, 0.0, 2.0/3.0}};
        // RK_beta = {1.0, 0.25, 2.0/3.0};
    }
    else {
        std::cerr << "Invalid RK_type value." << std::endl;
        exit(0);
    }
    
    CFL = 0.5; // Courant number
    
    // allocate memory
    U = vec3d(RK_stage, vec2d(num_var, vec1d(NN, 0.0))); // conservative variables
    spatial_discre = 1; // spatial discretization method (1: divergence of flux tensor, 2: wave-propagation method)
    if (dim >= 1){
        W_x_L = vec3d(BVD, vec2d(num_var, vec1d(NBX, 0.0))); // reconstructed left-side cell boundary value in x-direction
        W_x_R = vec3d(BVD, vec2d(num_var, vec1d(NBX, 0.0))); // reconstructed right-side cell boundary value in x-direction
        F_x = vec2d(num_var, vec1d(NBX, 0.0)); // flux vector in x-direction
        Apdq_x = vec2d(num_var, vec1d(NBX, 0.0)); // positive fluctuation in x-direction
        Amdq_x = vec2d(num_var, vec1d(NBX, 0.0)); // negative fluctuation in x-direction
        Adq_x = vec2d(num_var, vec1d(NN, 0.0)); // fluctuation in x-direction
    }
    if (dim >= 2){
        W_y_L = vec3d(BVD, vec2d(num_var, vec1d(NBY, 0.0))); // reconstructed left-side cell boundary value in y-direction
        W_y_R = vec3d(BVD, vec2d(num_var, vec1d(NBY, 0.0))); // reconstructed right-side cell boundary value in y-direction
        F_y = vec2d(num_var, vec1d(NBY, 0.0)); // flux vector in y-direction
        Apdq_y = vec2d(num_var, vec1d(NBY, 0.0)); // positive fluctuation in y-direction
        Amdq_y = vec2d(num_var, vec1d(NBY, 0.0)); // negative fluctuation in y-direction
        Adq_y = vec2d(num_var, vec1d(NN, 0.0)); // fluctuation in y-direction
    }
    if (dim >= 3){
        W_z_L = vec3d(BVD, vec2d(num_var, vec1d(NBZ, 0.0))); // reconstructed left-side cell boundary value in z-direction
        W_z_R = vec3d(BVD, vec2d(num_var, vec1d(NBZ, 0.0))); // reconstructed right-side cell boundary value in z-direction
        F_z = vec2d(num_var, vec1d(NBZ, 0.0)); // flux vector in z-direction
        Apdq_z = vec2d(num_var, vec1d(NBZ, 0.0)); // positive fluctuation in z-direction
        Amdq_z = vec2d(num_var, vec1d(NBZ, 0.0)); // negative fluctuation in z-direction
        Adq_z = vec2d(num_var, vec1d(NN, 0.0)); // fluctuation in z-direction
    }
    VOF_x_Riemann=vec1d(NBX,0.0);
    u_Riemann=vec1d(NBX,0.0);
    VOF_y_Riemann=vec1d(NBY,0.0);
    v_Riemann=vec1d(NBY,0.0);
    VOF_z_Riemann=vec1d(NBZ,0.0);
    w_Riemann=vec1d(NBZ,0.0);
    BVD_active = vec4i(dim, vec3i(RK_stage, vec2i(num_var, vec1i(NN, 0)))); // record cells where MUSCL has been replaced by THINC
    
    // surface tension parameters
    sigma_CSF=0.0;
    order_grad_VOF=0;
    if (surface_tension_type >= 1 || 1){
        sigma_CSF=1.0; // surface tension coefficient
        order_grad_VOF=2; // order of gradient of VOF for surface tension calculation
        
        ngx_CSF=max(order_grad_VOF/2,ngx); // number of ghost cells for CSF model in x-direction
        if (dim >= 2) ngy_CSF=ngx_CSF; // number of ghost cells for CSF model in y-direction
        else ngy_CSF=0; // number of ghost cells for CSF model in y-direction
        if (dim >= 3) ngz_CSF=ngx_CSF; // number of ghost cells for CSF model in z-direction
        else ngz_CSF=0; // number of ghost cells for CSF model in z-direction
        NX_CSF=nx+ngx_CSF*2; // number of cell in x-direction including ghost cell for CSF
        NY_CSF=ny+ngy_CSF*2; // number of cell in y-direction including ghost cell for CSF
        NZ_CSF=nz+ngz_CSF*2; // number of cell in z-direction including ghost cell for CSF
        NN_CSF=NX_CSF*NY_CSF*NZ_CSF; // total number of cell including ghost cell for CSF
        
        grad_VOF=vec2d(3,vec1d(NN_CSF)); // gradient of VOF at cell center in x, y, z-direction
        normal_vec=vec2d(4,vec1d(NN_CSF)); // 4:normal vector existence
        curv=vec1d(NN_CSF);
        coefficient_linear_polynoimal_1stDerivative_CellCenter(order_grad_VOF+1);
        
        // filtering for surface tension
        if (surface_tension_type==2){
            weight_filter=vec1d(NN,0.0);
            weight_sum_filter=vec1d(NN,0.0);
            curv_sum_filter=vec1d(NN_CSF);
        }
        
        // for debug
        F_CSF = vec2d(3, vec1d(NN, 0.0)); // surface tension force vector in x, y, z-direction for CSF model
    }
    
    // for THINC scheme
    beta = 1.6; // gradient parameter
    T1_THINC = tanh(beta / 2.0); // T1 should be precalculated for reducing computational cost
}

// Read parameters from file (ignores comment lines starting with #)
std::unordered_map<std::string, std::string> read_params(const std::string& filename) {
    std::unordered_map<std::string, std::string> params;
    std::ifstream infile(filename);
    std::string line;

    while (std::getline(infile, line)) {
        // Remove leading/trailing whitespace
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);

        // Skip empty or comment lines
        if (line.empty() || line[0] == '#') continue;

        std::istringstream iss(line);
        std::string key, value;
        if (std::getline(iss, key, '=') && std::getline(iss, value)) {
            key.erase(0, key.find_first_not_of(" \t"));
            key.erase(key.find_last_not_of(" \t") + 1);
            value.erase(0, value.find_first_not_of(" \t"));
            value.erase(value.find_last_not_of(" \t") + 1);
            // params[key] = value;
            if (!key.empty() && !value.empty()) {
                params[key] = value;
            }
        }
    }
    return params;
}

void EOS_param(){
    
    if (material=="water-air1_nondim"){
        EOS_type=2;
        gamma1=4.4; gamma2=1.4;
        pi1=6000.0; pi2=0.0;
        eta1=0.0; eta2=0.0;
        eta1d=0.0; eta2d=0.0;
        Cv1=0.0; Cv2=0.0;
    }
    else if (material=="water-air2_nondim"){
        EOS_type=2;
        gamma1=5.5; gamma2=1.4;
        pi1=1.505; pi2=0.0;
        eta1=0.0; eta2=0.0;
        eta1d=0.0; eta2d=0.0;
        Cv1=0.0; Cv2=0.0;
    }
    else {
        cout<<"material is not defined."<<endl;
        getchar();
    }
}

void EOS_param_ref
(
    double *Gammak, double *p_refk, double *e_refk, 
    double rhok, double gammak, double pik, double etak,
    double cB1k, double cB2k, double cE1k, double cE2k, double rho0k, double e0k
){
    
    //ideal gas
    if (EOS_type==1){
        *Gammak=gammak-1.0;
        *p_refk=0.0;
        *e_refk=0.0;
    }
    
    //stiffened gas
    else if (EOS_type==2){
        *Gammak=gammak-1.0;
        *p_refk=-gammak*pik;
        *e_refk=etak;
    }
    
    //Mie-Gruneisen EOS
    else if (EOS_type==3){
        double rho_ratio=rho0k/rhok;
        *Gammak=gammak-1.0;
        *p_refk=cB1k*pow(rho_ratio,-cE1k)-cB2k*pow(rho_ratio,-cE2k);
        *e_refk=-cB1k/((1.0-cE1k)*rho0k)*(pow(rho_ratio,1.0-cE1k)-1.0)+cB2k/((1.0-cE2k)*rho0k)*(pow(rho_ratio,1.0-cE2k)-1.0)-e0k;
    }
}

double sound_speed_square
(
    double rhok, double pk,
    double gammak, double pik, double etak,
    double cB1k, double cB2k, double cE1k, double cE2k, double rho0k, double e0k
){
    
    double Gammak,p_refk,e_refk,dGammak,dp_refk,de_refk,cck;
    
    EOS_param_ref(&Gammak,&p_refk,&e_refk,rhok,gammak,pik,etak,cB1k,cB2k,cE1k,cE2k,rho0k,e0k);
    if (EOS_type==1 || EOS_type==2){
        dGammak=0.0;
        dp_refk=0.0;
        de_refk=0.0;
    }
    else if (EOS_type==3){
        double rho_ratio=rhok/rho0k;
        dGammak=0.0;
        dp_refk=(cE1k*cB1k*pow(rho_ratio,cE1k-1.0)-cE2k*cB2k*pow(rho_ratio,cE2k-1.0))/rho0k;
        de_refk=(cB1k*pow(rho_ratio,cE1k-2.0)-cB2k*pow(rho_ratio,cE2k-2.0))/pow_int(rho0k,2);
    }
    cck=((Gammak+1.0)*pk-p_refk)/rhok+(pk-p_refk)/Gammak*dGammak+dp_refk-rhok*Gammak*de_refk;
    return cck;
}

void initial_condition(){
    
    if (dim == 1){
        // 1D shock-tube problem
        if (problem_type == 1){
            t_end = 0.2; // time to end calculation
            initial_condition_1D_GasLiquidRiemann();
        }
        else if (problem_type == 2){
            
        }
    }
    else if (dim == 2){
        // 2D static droplet problem
        if (problem_type == 1){
            t_end = 10.0; // time to end calculation
            initial_condition_2D_static_droplet();
        }
    }
    else if (dim == 3){
        // 3D static droplet problem
        if (problem_type == 1){
            
        }
    }
}

void initial_condition_1D_GasLiquidRiemann(){
    int i,j,k,Ic;
    double alpha1,alpha1rho1,alpha2rho2,rhou,rhov,rhow,rhoE,rho1,rho2,u,v,w,p;
    double x_border=0.0;
    
    for (i = ngx; i < ngx + nx; i++){
        for (j = ngy; j < ngy + ny; j++){
            for (k = ngz; k < ngz + nz; k++){
                
                if (xc[i]<x_border){
                    alpha1=1.0e-6;
                    rho1=0.991;
                    rho2=1.241;
                    u=v=w=0.0;
                    p=2.753;
                }
                else {
                    alpha1=1.0-1.0e-6;
                    rho1=0.991;
                    rho2=1.241;
                    u=v=w=0.0;
                    p=3.059e-4;
                }
                
                prim_to_cons_5eq(&alpha1rho1,&alpha2rho2,&rhou,&rhov,&rhow,&rhoE,
                            alpha1,rho1,rho2,u,v,w,p);
                
                Ic=I_c(i,j,k);
                U[0][0][Ic]=alpha1;
                U[0][1][Ic]=alpha1rho1;
                U[0][2][Ic]=alpha2rho2;
                U[0][3][Ic]=rhou;
                U[0][4][Ic]=rhov;
                U[0][5][Ic]=rhow;
                U[0][6][Ic]=rhoE;
            }
        }
    }
    
}

void initial_condition_2D_static_droplet(){
    int i, j, k, Ic;
    double alpha1, alpha1rho1, alpha2rho2, rhou, rhov, rhow, rhoE, rho1, rho2, u, v, w, p;
    double x_center, y_center, r_droplet, x_r, y_r, r, h = fmin(dx, dy);
    
    x_center=0.0, y_center=0.0, r_droplet=1.0;
    
    
    for (i = ngx; i < ngx + nx; i++){
        for (j = ngy; j < ngy + ny; j++){
            for (k = ngz; k < ngz + nz; k++){
                Ic = I_c(i, j, k);
                
                x_r = xc[i] - x_center;
                y_r = yc[j] - y_center;
                r = sqrt(x_r * x_r + y_r * y_r);
                alpha1 = 1.0 / (1.0 + exp((r - r_droplet) / (1.0 * 0.72 * h)));
                alpha1 = fmax(1.0e-8, fmin(1.0 - 1.0e-8, alpha1));
                rho1 = 1000.0;
                rho2 = 1.0;
                u = v = w = 0.0;
                p = 1.0 + alpha1;
                
                prim_to_cons_5eq(&alpha1rho1, &alpha2rho2, &rhou, &rhov, &rhow, &rhoE, 
                                alpha1, rho1, rho2, u, v, w, p);
                
                U[0][0][Ic] = alpha1;
                U[0][1][Ic] = alpha1rho1;
                U[0][2][Ic] = alpha2rho2;
                U[0][3][Ic] = rhou;
                U[0][4][Ic] = rhov;
                U[0][5][Ic] = rhow;
                U[0][6][Ic] = rhoE;
            }
        }
    }
        
    //symmetry preserving for y-axis
    // for (i=0;i<nx;i++){
    //     for (j=0;j<ny/2;j++){
    //         for (k=0;k<nz;k++){
    //             for (m=0;m<num_var;m++){
    //                 U[0][m][I(ng+i,ng+ny-1-j,ng+k)]=U[0][m][I(ng+i,ng+j,ng+k)];
    //             }
    //         }
    //     }
    // }
}

void prim_to_cons_5eq(double *alpha1rho1,double *alpha2rho2,double *rhou,double *rhov,double *rhow,double *rhoE,
                  double alpha1,double rho1,double rho2,double u,double v,double w,double p){
    double alpha2,rho_mix,udotu;
    double Gamma1,p_ref1,e_ref1,Gamma2,p_ref2,e_ref2,Gamma,p_ref,rhoe_ref;
    
    alpha2=1.0-alpha1;
    *alpha1rho1=alpha1*rho1;
    *alpha2rho2=alpha2*rho2;
    rho_mix=(*alpha1rho1)+(*alpha2rho2);
    *rhou=rho_mix*u;
    *rhov=rho_mix*v;
    *rhow=rho_mix*w;
    udotu=q2(u,v,w);
    *rhoE=alpha1*((p+gamma1*pi1)/(gamma1-1.0)+rho1*eta1+0.5*rho1*udotu)+alpha2*((p+gamma2*pi2)/(gamma2-1.0)+rho2*eta2+0.5*rho2*udotu);
}

void cons_to_prim_5eq(double *rho1,double *rho2,double *u,double *v,double *w,double *p,
                  double alpha1,double alpha1rho1,double alpha2rho2,double rhou,double rhov,double rhow,double rhoE){
    double alpha2,rho_mix,udotu;
    double Gamma1,p_ref1,e_ref1,Gamma2,p_ref2,e_ref2,Gamma,p_ref,rhoe_ref;
    
    alpha2=1.0-alpha1;
    *rho1=alpha1rho1/alpha1;
    *rho2=alpha2rho2/alpha2;
    rho_mix=alpha1rho1+alpha2rho2;
    *u=rhou/rho_mix;
    *v=rhov/rho_mix;
    *w=rhow/rho_mix;
    udotu=q2(*u,*v,*w);
    *p=((rhoE-0.5*rho_mix*udotu)-(alpha1rho1*eta1+alpha2rho2*eta2)-(alpha1*gamma1*pi1/(gamma1-1.0)+alpha2*gamma2*pi2/(gamma2-1.0)))/(alpha1/(gamma1-1.0)+alpha2/(gamma2-1.0));
}

void initialize_BVD_active(){
    int i, j, k, m, Ic, d, rk;
    for (d = 0; d < dim; d++){
        for (rk = 0; rk < RK_stage; rk++){
            for (m = 0; m < num_var; m++){
                for (i = 0; i < NX; i++){
                    for (j = 0; j < NY; j++){
                        for (k = 0; k < NZ; k++){
                            Ic = I_c(i, j, k);
                            BVD_active[d][rk][m][Ic] = 0; // initialize BVD_active
                        }
                    }
                }
            }
        }
    }
}

void boundary_condition(int rk){
    int i, j, k, m, Ic, Ic_ref;
    
    // boundary condition in x-direction
    if (dim >= 1){
        for (k = ngz; k < ngz + nz; k++){ // each cell in z-direction
            for (j = ngy; j < ngy + ny; j++){ // each cell in y-direction
                for (i = 0; i < ngx; i++){ // ghost cells in x-direction
                    
                    // negative side from x_range[0]
                    Ic = I_c(i, j, k);
                    if (boundary_type[0] == 1){ // outflow boundary condition
                        Ic_ref = I_c(ngx, j, k); // reference cell index
                        for (m = 0; m < num_var; m++){
                            U[rk][m][Ic] = U[rk][m][Ic_ref];
                        }
                    }
                    else if (boundary_type[0] == 2){ // reflective boundary condition
                        Ic_ref = I_c(ngx * 2 - 1 - i, j, k); // reference cell index
                        for (m = 0; m < num_var; m++){
                            U[rk][m][Ic] = U[rk][m][Ic_ref];
                        }
                        U[rk][3][Ic] = -U[rk][3][Ic]; // invert velocity component
                    }
                    else if (boundary_type[0] == 3){ // periodic boundary condition
                        Ic_ref = I_c(nx + i, j, k); // reference cell index
                        for (m = 0; m < num_var; m++){
                            U[rk][m][Ic] = U[rk][m][Ic_ref];
                        }
                    }
                    else if (boundary_type[0] == 4){ //
                        
                    }
                    
                    // positive side from x_range[1]
                    Ic = I_c(NX - 1 - i, j, k);
                    if (boundary_type[1] == 1){ // outflow boundary condition
                        Ic_ref = I_c(nx + ngx - 1, j, k); // reference cell index
                        for (m = 0; m < num_var; m++){
                            U[rk][m][Ic] = U[rk][m][Ic_ref];
                        }
                    }
                    else if (boundary_type[1] == 2){ // reflective boundary condition
                        Ic_ref = I_c(nx + i, j, k); // reference cell index
                        for (m = 0; m < num_var; m++){
                            U[rk][m][Ic] = U[rk][m][Ic_ref];
                        }
                        U[rk][3][Ic] = -U[rk][3][Ic]; // invert velocity component
                    }
                    else if (boundary_type[1] == 3){ // periodic boundary condition
                        Ic_ref = I_c(ngx * 2 - 1 - i, j, k); // reference cell index
                        for (m = 0; m < num_var; m++){
                            U[rk][m][Ic] = U[rk][m][Ic_ref];
                        }
                    }
                }
            }
        }
    }
    
    // boundary condition in y-direction
    if (dim >= 2){
        for (k = ngz; k < ngz + nz; k++){ // each cell in z-direction
            for (j = 0; j < ngy; j++){ // each cell in y-direction
                for (i = ngx; i < ngx + nx; i++){ // ghost cells in x-direction
                    
                    // negative side from y_range[0]
                    Ic = I_c(i, j, k);
                    if (boundary_type[2] == 1){ // outflow boundary condition
                        Ic_ref = I_c(i, ngy, k); // reference cell index
                        for (m = 0; m < num_var; m++){
                            U[rk][m][Ic] = U[rk][m][Ic_ref];
                        }
                    }
                    else if (boundary_type[2] == 2){ // reflective boundary condition
                        Ic_ref = I_c(i, ngy * 2 - 1 - j, k); // reference cell index
                        for (m = 0; m < num_var; m++){
                            U[rk][m][Ic] = U[rk][m][Ic_ref];
                        }
                        U[rk][4][Ic] = -U[rk][4][Ic]; // invert velocity component
                    }
                    else if (boundary_type[2] == 3){ // periodic boundary condition
                        Ic_ref = I_c(i, ny + j, k); // reference cell index
                        for (m = 0; m < num_var; m++){
                            U[rk][m][Ic] = U[rk][m][Ic_ref];
                        }
                    }
                    else if (boundary_type[2] == 4){ //
                        
                    }
                    
                    // positive side from y_range[1]
                    Ic = I_c(i, NY - 1 - j, k);
                    if (boundary_type[3] == 1){ // outflow boundary condition
                        Ic_ref = I_c(i, ny + ngy - 1, k); // reference cell index
                        for (m = 0; m < num_var; m++){
                            U[rk][m][Ic] = U[rk][m][Ic_ref];
                        }
                    }
                    else if (boundary_type[3] == 2){ // reflective boundary condition
                        Ic_ref = I_c(i, ny + j, k); // reference cell index
                        for (m = 0; m < num_var; m++){
                            U[rk][m][Ic] = U[rk][m][Ic_ref];
                        }
                        U[rk][4][Ic] = -U[rk][4][Ic]; // invert velocity component
                    }
                    else if (boundary_type[3] == 3){ // periodic boundary condition
                        Ic_ref = I_c(i, ngy * 2 - 1 - j, k); // reference cell index
                        for (m = 0; m < num_var; m++){
                            U[rk][m][Ic] = U[rk][m][Ic_ref];
                        }
                    }
                }
            }
        }
    }
    
    // boundary condition in z-direction
    if (dim >= 3){
        for (k = 0; k < ngz; k++){ // each cell in z-direction
            for (j = ngy; j < ngy + ny; j++){ // each cell in y-direction
                for (i = ngx; i < ngx + nx; i++){ // ghost cells in x-direction
                    
                    // negative side from z_range[0]
                    Ic = I_c(i, j, k);
                    if (boundary_type[4] == 1){ // outflow boundary condition
                        Ic_ref = I_c(i, j, ngz); // reference cell index
                        for (m = 0; m < num_var; m++){
                            U[rk][m][Ic] = U[rk][m][Ic_ref];
                        }
                    }
                    else if (boundary_type[4] == 2){ // reflective boundary condition
                        Ic_ref = I_c(i, j, ngz * 2 - 1 - k); // reference cell index
                        for (m = 0; m < num_var; m++){
                            U[rk][m][Ic] = U[rk][m][Ic_ref];
                        }
                        U[rk][5][Ic] = -U[rk][5][Ic]; // invert velocity component
                    }
                    else if (boundary_type[4] == 3){ // periodic boundary condition
                        Ic_ref = I_c(i, j, nz + k); // reference cell index
                        for (m = 0; m < num_var; m++){
                            U[rk][m][Ic] = U[rk][m][Ic_ref];
                        }
                    }
                    else if (boundary_type[4] == 4){ //
                        
                    }
                    
                    // positive side from y_range[1]
                    Ic = I_c(i, j, NZ - 1 - k);
                    if (boundary_type[5] == 1){ // outflow boundary condition
                        Ic_ref = I_c(i, j, nz + ngz - 1); // reference cell index
                        for (m = 0; m < num_var; m++){
                            U[rk][m][Ic] = U[rk][m][Ic_ref];
                        }
                    }
                    else if (boundary_type[5] == 2){ // reflective boundary condition
                        Ic_ref = I_c(i, j, nz + k); // reference cell index
                        for (m = 0; m < num_var; m++){
                            U[rk][m][Ic] = U[rk][m][Ic_ref];
                        }
                        U[rk][5][Ic] = -U[rk][5][Ic]; // invert velocity component
                    }
                    else if (boundary_type[5] == 3){ // periodic boundary condition
                        Ic_ref = I_c(i, j, ngz * 2 - 1 - k); // reference cell index
                        for (m = 0; m < num_var; m++){
                            U[rk][m][Ic] = U[rk][m][Ic_ref];
                        }
                    }
                }
            }
        }
    }
}

void reconstruction_dim(int d, int rk, vec3d& W_L, vec3d& W_R){
    int i, j, k, m, xi, Ip, Im;
    
    vec1i loop_begin = loop_begin_base;
    vec1i loop_size = loop_size_base;
    loop_begin[d] -= 1 + BVD_s;
    loop_size[d]  += (1 + BVD_s) * 2;
    
    // reconstruction in d-direction
    #pragma omp parallel
    {
        double q_L, q_R;
        vec2d stencil(num_var, vec1d(ns));
        vec1i idx(3);
        
        #pragma omp for collapse(3) private(i, j, k, m, xi, Ip, Im)
        for (k = loop_begin[2]; k < loop_begin[2] + loop_size[2]; k++){ // each cell in z-direction
            for (j = loop_begin[1]; j < loop_begin[1] + loop_size[1]; j++){ // each cell in y-direction
                for (i = loop_begin[0]; i < loop_begin[0] + loop_size[0]; i++){ // each cell in x-direction
                    
                    // put conservative variables in stencil
                    for (m = 0; m < num_var; m++){ // each variables
                        idx = {i, j, k};
                        idx[d] = idx[d] - (ns - 1) / 2;
                        for (xi = 0; xi < ns; xi++){
                            // stencil[m][xi] = U[rk][m][I_c(i - (ns - 1) / 2 + xi, j, k)]; // stencil for reconstruction
                            stencil[m][xi] = U[rk][m][I_c(idx[0], idx[1], idx[2])];
                            idx[d]++;
                        }
                    }
                    
                    // calculate primitive variables from conservative variables
                    for (xi = 0; xi < ns; xi++){
                        cons_to_prim_5eq(&stencil[1][xi], &stencil[2][xi], &stencil[3][xi], &stencil[4][xi], &stencil[5][xi], &stencil[6][xi],
                            stencil[0][xi], stencil[1][xi], stencil[2][xi], stencil[3][xi], stencil[4][xi], stencil[5][xi], stencil[6][xi]);
                        stencil[1][xi] = stencil[0][xi] * stencil[1][xi]; // alpha1rho1=alpha1*rho1
                        stencil[2][xi] = (1.0 - stencil[0][xi]) * stencil[2][xi]; // alpha2rho2=(1-alpha1)*rho2
                    }
                        
                    // calculate cell boundary values
                    // q_L: left-side cell boundary value at x_{i+1/2}
                    // q_R: right-side cell boundary value at x_{i-1/2}
                    // Ip = I_x(i + 1, j, k); // index of cell boundary at x_{i+1/2}
                    // Im = I_x(i, j, k); // index of cell boundary at x_{i-1/2}
                    get_cb_indices_from_cc(d, i, j, k, Im, Ip); // get cell boundary indices
                    for (m = 0; m < num_var; m++){
                        
                        if (scheme_type == 1){
                            MUSCL(&q_L, &q_R, stencil[m]);
                            W_L[0][m][Ip] = q_L;
                            W_R[0][m][Im] = q_R;
                        }
                        else if (scheme_type == 2){
                            THINC(&q_L, &q_R, stencil[m]);
                            W_L[0][m][Ip] = q_L;
                            W_R[0][m][Im] = q_R;
                        }
                        else if (scheme_type == 3){
                            MUSCL(&q_L, &q_R, stencil[m]);
                            W_L[0][m][Ip] = q_L;
                            W_R[0][m][Im] = q_R;
                            THINC(&q_L, &q_R, stencil[m]);
                            W_L[1][m][Ip] = q_L;
                            W_R[1][m][Im] = q_R;
                        }
                        
                    }
                }
            }
        }
    }
    
    if (BVD > 1){ // MUSCL or THINC scheme is selected at each cell following BVD selection algorithm
        // BVD_selection_x(rk);
        BVD_selection_dim(d, rk, W_L, W_R, BVD_active[d]);
    }
    
    loop_begin = loop_begin_base;
    loop_size = loop_size_base;
    loop_begin[d] -= 1;
    loop_size[d]  += 2;
    #pragma omp parallel for collapse(3) private(i, j, k, m, Ip, Im)
    for (k = loop_begin[2]; k < loop_begin[2] + loop_size[2]; k++){ // each cell in z-direction
        for (j = loop_begin[1]; j < loop_begin[1] + loop_size[1]; j++){ // each cell in y-direction
            for (i = loop_begin[0]; i < loop_begin[0] + loop_size[0]; i++){ // each cell in x-direction
                // Ip = I_x(i + 1, j, k); // index of cell boundary at x_{i+1/2}
                // Im = I_x(i, j, k); // index of cell boundary at x_{i-1/2}
                get_cb_indices_from_cc(d, i, j, k, Im, Ip); // get cell boundary indices
                
                W_L[0][1][Ip] = W_L[0][1][Ip] / W_L[0][0][Ip]; //rho1=alpha1rho1/alpha1
                W_L[0][2][Ip] = W_L[0][2][Ip] / (1.0 - W_L[0][0][Ip]); //rho2=alpha2rho2/alpha2
                
                W_R[0][1][Im] = W_R[0][1][Im] / W_R[0][0][Im]; //rho1=alpha1rho1/alpha1
                W_R[0][2][Im] = W_R[0][2][Im] / (1.0 - W_R[0][0][Im]); //rho2=alpha2rho2/alpha2
            }
        }
    }
    
    // bvc_Euler_x(0); // if the reconstructed values violate positivity, the reconstruction function is degraded to 1st-order method to keep positivity.
    
}

void MUSCL(double *qL, double *qR, const vec1d& q){
    
    double Phi_L, Phi_R, r_i, dq;
    
    if (fabs(q[1] - q[0]) > eps && fabs(q[2] - q[1]) > eps){ // avoid zero-division
        r_i = (q[1] - q[0]) / (q[2] - q[1]);
        Phi_L = Phi_MUSCL(r_i);
        dq = 0.5 * Phi_L * (q[2] - q[1]);
        *qL = q[1] + dq; // left-side cell boundary value
        *qR = q[1] - dq; // right-side cell boundary value
    }
    else {
        *qL = q[1];
        *qR = q[1];
    }

}

double Phi_MUSCL(double r_i){
    return (r_i + fabs(r_i)) / (1.0 + fabs(r_i)); // van leer limiter
    // return 0; // piecewise-constant reconstruction
}

void THINC(double *qL, double *qR, const vec1d& q){
    double q_p, q_m, alpha, eps_THINC = 1e-15, T1, T2;
    
    T1 = T1_THINC;
    
    if ((q[1] - q[0]) * (q[2] - q[1]) > eps_THINC){
        q_p = 0.5 * (q[0] + q[2]);
        q_m = 0.5 * (q[2] - q[0]);
        alpha = (q[1] - q_p) / q_m;
        T2 = tanh(0.5 * alpha * beta);
        *qL = q_p + q_m * (T1 + T2 / T1) / (1.0 + T2); // left-side cell boundary value
        *qR = q_p - q_m * (T1 - T2 / T1) / (1.0 - T2); // right-side cell boundary value
    }
    else {
        *qL = q[1];
        *qR = q[1];
    }
}

void BVD_selection_dim(int d, int rk, vec3d& W_L, vec3d& W_R, vec3i& BVD_active_dim){
    int i, j, k, m, Im, Ip;
    double TBV[2];
    
    vec1i loop_begin = loop_begin_base;
    vec1i loop_size = loop_size_base;
    loop_begin[d] -= BVD_s;
    loop_size[d]  += BVD_s * 2;

    // calculate and compare TBV (Total Boundary Variation)
    #pragma omp parallel for collapse(4) private(k,j,i,Im,Ip,TBV)
    for (m = 0; m < num_var; m++) {
        for (k = loop_begin[2]; k < loop_begin[2] + loop_size[2]; k++) { // each cell in z-direction
            for (j = loop_begin[1]; j < loop_begin[1] + loop_size[1]; j++) { // each cell in y-direction
                for (i = loop_begin[0]; i < loop_begin[0] + loop_size[0]; i++) { // each cell in x-direction
                    get_cb_indices_from_cc(d, i, j, k, Im, Ip); // get cell boundary indices
                    TBV[0] = fabs(W_L[0][m][Im] - W_R[0][m][Im]) + fabs(W_L[0][m][Ip] - W_R[0][m][Ip]); // TBV of MUSCL
                    TBV[1] = fabs(W_L[1][m][Im] - W_R[1][m][Im]) + fabs(W_L[1][m][Ip] - W_R[1][m][Ip]); // TBV of THINC
                    if (TBV[0] > TBV[1]){
                        BVD_active_dim[rk][m][I_c(i, j, k)] = 1;
                    }
                }
            }
        }
    }
    
    // select numerical scheme which has smaller TBV value
    #pragma omp parallel for collapse(4) private(k,j,i,Im,Ip)
    for (m = 0; m < num_var; m++){
        for (k = loop_begin[2]; k < loop_begin[2] + loop_size[2]; k++) { // each cell in z-direction
            for (j = loop_begin[1]; j < loop_begin[1] + loop_size[1]; j++) { // each cell in y-direction
                for (i = loop_begin[0]; i < loop_begin[0] + loop_size[0]; i++) { // each cell in x-direction
                    if (BVD_active_dim[rk][m][I_c(i, j, k)] == 1){
                        get_cb_indices_from_cc(d, i, j, k, Im, Ip); // get cell boundary indices
                        W_L[0][m][Ip] = W_L[1][m][Ip];
                        W_R[0][m][Im] = W_R[1][m][Im];
                    }
                }
            }
        }
    }
}

void Riemann_solver_5eq_HLLC_dim(int d, double *MWS_d, const vec3d& W_L, const vec3d& W_R, vec2d& F, vec2d& Amdq, vec2d& Apdq, vec2d& Adq, vec1d& vel_Riemann, vec1d& col_Riemann){
    int i,j,k,m;
    double mws_d=0.0;
    
    //  Riemann solver in d-direction
    #pragma omp parallel
    {
        int Ib, Ibm, Ibp, Ic, Icm, Icp;
        double alpha1_L, alpha2_L, rho1_L, rho2_L, u_L, v_L, w_L, S_L, Y1_L, rho_L, p_L, c_L, S_ratio_L;
        double alpha1rho1_L, alpha2rho2_L, rhou_L, rhov_L, rhow_L, rhoE_L, cc1_L, cc2_L;
        vec1d U_L(num_var), F_L(num_var), U_Lstar(num_var);
        double alpha1_R, alpha2_R, rho1_R, rho2_R, u_R, v_R, w_R, S_R, Y1_R, rho_R, p_R, c_R, S_ratio_R;
        double alpha1rho1_R, alpha2rho2_R, rhou_R, rhov_R, rhow_R, rhoE_R, cc1_R, cc2_R;
        vec1d U_R(num_var), F_R(num_var), U_Rstar(num_var);
        double S_star;
        double gamma_mix, p_ref;
        
        vec1i loop_offset = {0, 0, 0};
        loop_offset[d] = 1; // offset for cell boundary indices
        
        vec1d velocity(3);
        int id0 = d, id1 = (d + 1) % 3, id2 = (d + 2) % 3; // indices for velocity components
        // vec1d Amdq_n(num_var), Apdq_n(num_var), Adq_n(num_var); // fluctuations at permutated bases
        vec1i m_list = {0, 1, 2, 3 + id0, 3 + id1, 3 + id2, 6}; // indices for variables
        
        double S_L_plus, S_L_minus, S_s_plus, S_s_minus, S_R_plus, S_R_minus, S_s_sign_plus, S_s_sign_minus;
        double W1, W2, W3;
        
        double kappa;
        
        // calculate fluctuation at cell boundary
        #pragma omp for collapse(3) private(k,j,i,m) reduction(max:mws_d)
        for (k = ngz; k < ngz + nz + loop_offset[2]; k++){ // each cell or cell-boundary in z-direction
            for (j = ngy; j < ngy + ny + loop_offset[1]; j++){ // each cell or cell-boundary in y-direction
                for (i = ngx; i < ngx + nx + loop_offset[0]; i++){ // each cell or cell-boundary in x-direction
                    // Ib=I_x(i,j,k); // index of cell boundary at x_{i-1/2}
                    switch (d) {
                        case 0:
                            Ib = I_x(i, j, k); // index of cell boundary at x_{i-1/2}
                            break;
                        case 1:
                            Ib = I_y(i, j, k); // index of cell boundary at y_{j-1/2}
                            break;
                        case 2:
                            Ib = I_z(i, j, k); // index of cell boundary at z_{k-1/2}
                            break;
                    }
                    
                    // left-side cell boundary value
                    alpha1_L=W_L[0][0][Ib];
                    alpha2_L=1.0-alpha1_L;
                    rho1_L=W_L[0][1][Ib];
                    rho2_L=W_L[0][2][Ib];
                    u_L=W_L[0][3][Ib];
                    v_L=W_L[0][4][Ib];
                    w_L=W_L[0][5][Ib];
                    p_L=W_L[0][6][Ib];
                    // rhoE_L=alpha1_L*((p_L+gamma1*pi1)/(gamma1-1.0)+rho1_L*eta1+0.5*rho1_L*q2(u_L,v_L,w_L))
                    //     +alpha2_L*((p_L+gamma2*pi2)/(gamma2-1.0)+rho2_L*eta2+0.5*rho2_L*q2(u_L,v_L,w_L));
                    prim_to_cons_5eq(&alpha1rho1_L,&alpha2rho2_L,&rhou_L,&rhov_L,&rhow_L,&rhoE_L,
                                    alpha1_L,rho1_L,rho2_L,u_L,v_L,w_L,p_L);
                    rho_L=alpha1_L*rho1_L+alpha2_L*rho2_L;
                    Y1_L=alpha1_L*rho1_L/rho_L;
                    if (sound_speed_type==1){
                        // c_L=sqrt(Y1_L*(gamma1*(p_L+pi1)/rho1_L)+(1.0-Y1_L)*(gamma2*(p_L+pi2)/rho2_L)); //frozen
                        // if (model==1) c_L=sqrt(1.0/(rho_L*(alpha1_L/(gamma1*(p_L+pi1))+alpha2_L/(gamma2*(p_L+pi2))))); //Wood
                        // else if (model==2) c_L=sqrt((Y1_L/(gamma1-1.0)*(gamma1*(p_L+pi1)/rho1_L)+(1.0-Y1_L)/(gamma2-1.0)*(gamma2*(p_L+pi2)/rho2_L))/(alpha1_L/(gamma1-1.0)+alpha2_L/(gamma2-1.0))); //Allaire
                        cc1_L=sound_speed_square(rho1_L,p_L,gamma1,pi1,eta1,cB11,cB21,cE11,cE21,rho01,e01);
                        cc2_L=sound_speed_square(rho2_L,p_L,gamma2,pi2,eta2,cB12,cB22,cE12,cE22,rho02,e02);
                        // c_L=sqrt((Y1_L/(gamma1-1.0)*(((Gamma1+1.0)*p_L-p_ref1)/rho1_L+(p_L-p_ref1)/Gamma1*dGamma1+dp_ref1-rho1_L*Gamma1*de_ref1)+(1.0-Y1_L)/(gamma2-1.0)*(((Gamma2+1.0)*p_L-p_ref2)/rho2_L+(p_L-p_ref2)/Gamma2*dGamma2+dp_ref2-rho2_L*Gamma2*de_ref2))/(alpha1_L/(gamma1-1.0)+alpha2_L/(gamma2-1.0)));
                        c_L=sqrt((Y1_L/(gamma1-1.0)*cc1_L+(1.0-Y1_L)/(gamma2-1.0)*cc2_L)/(alpha1_L/(gamma1-1.0)+alpha2_L/(gamma2-1.0)));
                    }
                    else if (sound_speed_type==2){
                        gamma_mix=1.0+1.0/(alpha1_L/(gamma1-1.0)+alpha2_L/(gamma2-1.0));
                        p_ref=(gamma_mix-1.0)/gamma_mix*(alpha1_L*pi1*gamma1/(gamma1-1.0)+alpha2_L*pi2*gamma2/(gamma2-1.0));
                        c_L=sqrt(gamma_mix*(p_L+p_ref)/rho_L);
                    }
                    // c_L=sqrt(alpha1_L*cc1_L+alpha2_L*cc2_L);
                    if (isnan(c_L) && 0){
                        printf("c_L is nan @HLLC\n");
                        printf("t=%f, t_step=%d, dim=%d, cell boundary:(%d,%d,%d), alpha1=%e, alpha2=%e, p=%e, rho=%e\n",t,t_step,d,i-ngx,j-ngy,k-ngz,alpha1_L,alpha2_L,p_L,rho_L);
                        getchar();
                    }
                    
                    // right-side cell boundary value
                    alpha1_R=W_R[0][0][Ib];
                    alpha2_R=1.0-alpha1_R;
                    rho1_R=W_R[0][1][Ib];
                    rho2_R=W_R[0][2][Ib];
                    u_R=W_R[0][3][Ib];
                    v_R=W_R[0][4][Ib];
                    w_R=W_R[0][5][Ib];
                    p_R=W_R[0][6][Ib];
                    // rhoE_R=alpha1_R*((p_R+gamma1*pi1)/(gamma1-1.0)+rho1_R*eta1+0.5*rho1_R*q2(u_R,v_R,w_R))
                    //     +alpha2_R*((p_R+gamma2*pi2)/(gamma2-1.0)+rho2_R*eta2+0.5*rho2_R*q2(u_R,v_R,w_R));
                    prim_to_cons_5eq(&alpha1rho1_R,&alpha2rho2_R,&rhou_R,&rhov_R,&rhow_R,&rhoE_R,
                                    alpha1_R,rho1_R,rho2_R,u_R,v_R,w_R,p_R);
                    //
                    rho_R=alpha1_R*rho1_R+alpha2_R*rho2_R;
                    Y1_R=alpha1_R*rho1_R/rho_R;
                    if (sound_speed_type==1){
                        // c_R=sqrt(Y1_R*(gamma1*(p_R+pi1)/rho1_R)+(1.0-Y1_R)*(gamma2*(p_R+pi2)/rho2_R)); //frozen
                        // if (model==1) c_R=sqrt(1.0/(rho_R*(alpha1_R/(gamma1*(p_R+pi1))+alpha2_R/(gamma2*(p_R+pi2))))); //Wood
                        // else if (model==2) c_R=sqrt((Y1_R/(gamma1-1.0)*(gamma1*(p_R+pi1)/rho1_R)+(1.0-Y1_R)/(gamma2-1.0)*(gamma2*(p_R+pi2)/rho2_R))/(alpha1_R/(gamma1-1.0)+alpha2_R/(gamma2-1.0))); //Allaire
                        cc1_R=sound_speed_square(rho1_R,p_R,gamma1,pi1,eta1,cB11,cB21,cE11,cE21,rho01,e01);
                        cc2_R=sound_speed_square(rho2_R,p_R,gamma2,pi2,eta2,cB12,cB22,cE12,cE22,rho02,e02);
                        // c_R=sqrt((Y1_R/(gamma1-1.0)*(((Gamma1+1.0)*p_R-p_ref1)/rho1_R+(p_R-p_ref1)/Gamma1*dGamma1+dp_ref1-rho1_R*Gamma1*de_ref1)+(1.0-Y1_R)/(gamma2-1.0)*(((Gamma2+1.0)*p_R-p_ref2)/rho2_R+(p_R-p_ref2)/Gamma2*dGamma2+dp_ref2-rho2_R*Gamma2*de_ref2))/(alpha1_R/(gamma1-1.0)+alpha2_R/(gamma2-1.0)));
                        c_R=sqrt((Y1_R/(gamma1-1.0)*cc1_R+(1.0-Y1_R)/(gamma2-1.0)*cc2_R)/(alpha1_R/(gamma1-1.0)+alpha2_R/(gamma2-1.0)));
                    }
                    else if (sound_speed_type==2){
                        gamma_mix=1.0+1.0/(alpha1_R/(gamma1-1.0)+alpha2_R/(gamma2-1.0));
                        p_ref=(gamma_mix-1.0)/gamma_mix*(alpha1_R*pi1*gamma1/(gamma1-1.0)+alpha2_R*pi2*gamma2/(gamma2-1.0));
                        c_R=sqrt(gamma_mix*(p_R+p_ref)/rho_R);
                    }
                    // c_R=sqrt(alpha1_R*cc1_R+alpha2_R*cc2_R);
                    if (isnan(c_R) && 0){
                        printf("c_R is nan @HLLC_x\n");
                        printf("t=%f, t_step=%d, dim=%d, cell boundary:(%d,%d,%d), alpha1=%e, alpha2=%e, p=%e, rho=%e\n",t,t_step,d,i-ngx,j-ngy,k-ngz,alpha1_R,alpha2_R,p_R,rho_R);
                        getchar();
                    }
                    
                    // permutation of bases of velocity vector
                    velocity = {u_L, v_L, w_L};
                    u_L = velocity[id0];
                    v_L = velocity[id1];
                    w_L = velocity[id2];
                    velocity = {u_R, v_R, w_R};
                    u_R = velocity[id0];
                    v_R = velocity[id1];
                    w_R = velocity[id2];
                    
                    // curvature at the cell boundary
                    switch (d) {
                        case 0:
                            Icm = I_c(i - 1, j, k);
                            break;
                        case 1:
                            Icm = I_c(i, j - 1, k);
                            break;
                        case 2:
                            Icm = I_c(i, j, k - 1);
                            break;
                    }
                    Icp = I_c(i, j, k); // index of cell center at (i,j,k)
                    if (surface_tension_type>=1) kappa=0.5*(curv[Icm]+curv[Icp]);
                    else kappa=0.0;
                    
                    // Direct Wave Speed Estimates
                    // S_L=min(u_L-c_L,u_R-c_R);
                    // S_R=max(u_L+c_L,u_R+c_R);
                    S_L=min({0.,u_L-c_L,(u_L+u_R)/2.-(c_L+c_R)/2.});
                    S_R=max({0.,u_R+c_R,(u_L+u_R)/2.+(c_L+c_R)/2.});
                    if (isnan(S_L) || isnan(S_R)) {
                        printf("SLR nan@xb"); getchar();
                    }
                    
                    // Max Wave Speed
                    mws_d=max({mws_d,fabs(S_L),fabs(S_R)});

                    // contact discontinuity speed
                    S_star=((p_R-p_L)+(rho_L*u_L*(S_L-u_L)-rho_R*u_R*(S_R-u_R))-sigma_CSF*kappa*(alpha1_R-alpha1_L))
                            /
                            (rho_L*(S_L-u_L)-rho_R*(S_R-u_R));

                    // HLLC solution
                    U_L[0]=alpha1_L;
                    U_L[1]=alpha1_L*rho1_L;
                    U_L[2]=alpha2_L*rho2_L;
                    U_L[3]=rho_L*u_L;
                    U_L[4]=rho_L*v_L;
                    U_L[5]=rho_L*w_L;
                    U_L[6]=rhoE_L;
                    F_L[0]=alpha1_L*u_L;
                    F_L[1]=alpha1_L*rho1_L*u_L;
                    F_L[2]=alpha2_L*rho2_L*u_L;
                    F_L[3]=p_L+rho_L*u_L*u_L;
                    F_L[4]=rho_L*v_L*u_L;
                    F_L[5]=rho_L*w_L*u_L;
                    F_L[6]=(rhoE_L+p_L)*u_L;
                    S_ratio_L=(S_L-u_L)/(S_L-S_star);
                    if      (spatial_discre == 1) U_Lstar[0]=alpha1_L*S_ratio_L;
                    else if (spatial_discre == 2) U_Lstar[0]=alpha1_L;
                    U_Lstar[1]=alpha1_L*rho1_L*S_ratio_L;
                    U_Lstar[2]=alpha2_L*rho2_L*S_ratio_L;
                    U_Lstar[3]=rho_L*S_star*S_ratio_L;
                    U_Lstar[4]=rho_L*v_L*S_ratio_L;
                    U_Lstar[5]=rho_L*w_L*S_ratio_L;
                    U_Lstar[6]=rho_L*S_ratio_L*(rhoE_L/rho_L+(S_star-u_L)*(S_star+(p_L-sigma_CSF*kappa*alpha1_L)/rho_L/(S_L-u_L)));
                    
                    U_R[0]=alpha1_R;
                    U_R[1]=alpha1_R*rho1_R;
                    U_R[2]=alpha2_R*rho2_R;
                    U_R[3]=rho_R*u_R;
                    U_R[4]=rho_R*v_R;
                    U_R[5]=rho_R*w_R;
                    U_R[6]=rhoE_R;
                    F_R[0]=alpha1_R*u_R;
                    F_R[1]=alpha1_R*rho1_R*u_R;
                    F_R[2]=alpha2_R*rho2_R*u_R;
                    F_R[3]=p_R+rho_R*u_R*u_R;
                    F_R[4]=rho_R*v_R*u_R;
                    F_R[5]=rho_R*w_R*u_R;
                    F_R[6]=(rhoE_R+p_R)*u_R;
                    S_ratio_R=(S_R-u_R)/(S_R-S_star);
                    if      (spatial_discre == 1) U_Rstar[0]=alpha1_R*S_ratio_R;
                    else if (spatial_discre == 2) U_Rstar[0]=alpha1_R;
                    U_Rstar[1]=alpha1_R*rho1_R*S_ratio_R;
                    U_Rstar[2]=alpha2_R*rho2_R*S_ratio_R;
                    U_Rstar[3]=rho_R*S_star*S_ratio_R;
                    U_Rstar[4]=rho_R*v_R*S_ratio_R;
                    U_Rstar[5]=rho_R*w_R*S_ratio_R;
                    U_Rstar[6]=rho_R*S_ratio_R*(rhoE_R/rho_R+(S_star-u_R)*(S_star+(p_R-sigma_CSF*kappa*alpha1_R)/rho_R/(S_R-u_R)));

                    // calculate fluctuation at cell boundary
                    S_L_plus=max(S_L,0.0);
                    S_L_minus=min(S_L,0.0);
                    S_s_plus=max(S_star,0.0);
                    S_s_minus=min(S_star,0.0);
                    S_R_plus=max(S_R,0.0);
                    S_R_minus=min(S_R,0.0);
                    S_s_sign_plus = 0.5 * (1.0 + sign(S_star));
                    S_s_sign_minus = 0.5 * (1.0 - sign(S_star));
                    if (spatial_discre==1){
                        for (m=0;m<num_var;m++){
                            F[m_list[m]][Ib] = S_s_sign_plus * (F_L[m] + S_L_minus * (U_Lstar[m] - U_L[m]))
                                             + S_s_sign_minus * (F_R[m] + S_R_plus * (U_Rstar[m] - U_R[m]));
                        }
                    }
                    else if (spatial_discre==2){
                        for (m=0;m<num_var;m++){
                            W1=U_Lstar[m]-U_L[m];
                            W2=U_Rstar[m]-U_Lstar[m];
                            W3=U_R[m]-U_Rstar[m];
                            Apdq[m_list[m]][Ib] = (S_L_plus * W1 + S_R_plus * W3) + S_s_plus * W2;
                            Amdq[m_list[m]][Ib] = (S_L_minus * W1 + S_R_minus * W3) + S_s_minus * W2;
                        }
                    }
                    
                    // calculate velocity and color function at cell boundary (for evaluating non-conservative and surface tension terms)
                    vel_Riemann[Ib] = S_s_sign_plus * (u_L + S_L_minus * (S_ratio_L - 1.0))
                                    + S_s_sign_minus * (u_R + S_R_plus * (S_ratio_R - 1.0));
                    col_Riemann[Ib] = S_s_sign_plus * alpha1_L * (1.0 + 0.5 * (1.0 - sign(S_L)) * (S_ratio_L - 1.0))
                                    + S_s_sign_minus * alpha1_R * (1.0 + 0.5 * (1.0 + sign(S_R)) * (S_ratio_R - 1.0));
                }
            }
        }
        
        // calculate fluctuation at cell inside
        if (spatial_discre==2){
            #pragma omp for collapse(3) private(k,j,i,m)
            for (k=ngz;k<ngz+nz;k++){ // each cell in z-direction
                for (j=ngy;j<ngy+ny;j++){ // each cell in y-direction
                    for (i=ngx;i<ngx+nx;i++){ // each cell in x-direction
                        // Ixm=I_x(i,j,k); // index of cell boundary at x_{i-1/2}
                        // Ixp=I_x(i+1,j,k); // index of cell boundary at x_{i+1/2}
                        get_cb_indices_from_cc(d, i, j, k, Ibm, Ibp); // get cell-boundary index
                        // Adq=s(1)W(1)+s(2)W(2)+s(3)W(3)
                        
                        // left-side cell boundary value
                        alpha1_L=W_R[0][0][Ibm];
                        alpha2_L=1.0-alpha1_L;
                        rho1_L=W_R[0][1][Ibm];
                        rho2_L=W_R[0][2][Ibm];
                        u_L=W_R[0][3][Ibm];
                        v_L=W_R[0][4][Ibm];
                        w_L=W_R[0][5][Ibm];
                        p_L=W_R[0][6][Ibm];
                        // rhoE_L=alpha1_L*((p_L+gamma1*pi1)/(gamma1-1.0)+rho1_L*eta1+0.5*rho1_L*q2(u_L,v_L,w_L))
                        //     +alpha2_L*((p_L+gamma2*pi2)/(gamma2-1.0)+rho2_L*eta2+0.5*rho2_L*q2(u_L,v_L,w_L));
                        prim_to_cons_5eq(&alpha1rho1_L,&alpha2rho2_L,&rhou_L,&rhov_L,&rhow_L,&rhoE_L,
                            alpha1_L,rho1_L,rho2_L,u_L,v_L,w_L,p_L);
                        rho_L=alpha1_L*rho1_L+alpha2_L*rho2_L;
                        Y1_L=alpha1_L*rho1_L/rho_L;
                        if (sound_speed_type==1){
                            // c_L=sqrt(Y1_L*(gamma1*(p_L+pi1)/rho1_L)+(1.0-Y1_L)*(gamma2*(p_L+pi2)/rho2_L)); //frozen
                            // if (model==1) c_L=sqrt(1.0/(rho_L*(alpha1_L/(gamma1*(p_L+pi1))+alpha2_L/(gamma2*(p_L+pi2))))); //Wood
                            // else if (model==2) c_L=sqrt((Y1_L/(gamma1-1.0)*(gamma1*(p_L+pi1)/rho1_L)+(1.0-Y1_L)/(gamma2-1.0)*(gamma2*(p_L+pi2)/rho2_L))/(alpha1_L/(gamma1-1.0)+alpha2_L/(gamma2-1.0))); //Allaire
                            cc1_L=sound_speed_square(rho1_L,p_L,gamma1,pi1,eta1,cB11,cB21,cE11,cE21,rho01,e01);
                            cc2_L=sound_speed_square(rho2_L,p_L,gamma2,pi2,eta2,cB12,cB22,cE12,cE22,rho02,e02);
                            c_L=sqrt((Y1_L/(gamma1-1.0)*cc1_L+(1.0-Y1_L)/(gamma2-1.0)*cc2_L)/(alpha1_L/(gamma1-1.0)+alpha2_L/(gamma2-1.0)));
                            // c_L=sqrt((Y1_L/(gamma1-1.0)*(((Gamma1+1.0)*p_L-p_ref1)/rho1_L+(p_L-p_ref1)/Gamma1*dGamma1+dp_ref1-rho1_L*Gamma1*de_ref1)+(1.0-Y1_L)/(gamma2-1.0)*(((Gamma2+1.0)*p_L-p_ref2)/rho2_L+(p_L-p_ref2)/Gamma2*dGamma2+dp_ref2-rho2_L*Gamma2*de_ref2))/(alpha1_L/(gamma1-1.0)+alpha2_L/(gamma2-1.0)));
                        }
                        else if (sound_speed_type==2){
                            gamma_mix=1.0+1.0/(alpha1_L/(gamma1-1.0)+alpha2_L/(gamma2-1.0));
                            p_ref=(gamma_mix-1.0)/gamma_mix*(alpha1_L*pi1*gamma1/(gamma1-1.0)+alpha2_L*pi2*gamma2/(gamma2-1.0));
                            c_L=sqrt(gamma_mix*(p_L+p_ref)/rho_L);
                        }
                        // c_L=sqrt(alpha1_L*cc1_L+alpha2_L*cc2_L);
                        
                        // right-side cell boundary value
                        alpha1_R=W_L[0][0][Ibp];
                        alpha2_R=1.0-alpha1_R;
                        rho1_R=W_L[0][1][Ibp];
                        rho2_R=W_L[0][2][Ibp];
                        u_R=W_L[0][3][Ibp];
                        v_R=W_L[0][4][Ibp];
                        w_R=W_L[0][5][Ibp];
                        p_R=W_L[0][6][Ibp];
                        // rhoE_R=alpha1_R*((p_R+gamma1*pi1)/(gamma1-1.0)+rho1_R*eta1+0.5*rho1_R*q2(u_R,v_R,w_R))
                        //     +alpha2_R*((p_R+gamma2*pi2)/(gamma2-1.0)+rho2_R*eta2+0.5*rho2_R*q2(u_R,v_R,w_R));
                        prim_to_cons_5eq(&alpha1rho1_R,&alpha2rho2_R,&rhou_R,&rhov_R,&rhow_R,&rhoE_R,
                            alpha1_R,rho1_R,rho2_R,u_R,v_R,w_R,p_R);
                        rho_R=alpha1_R*rho1_R+alpha2_R*rho2_R;
                        Y1_R=alpha1_R*rho1_R/rho_R;
                        if (sound_speed_type==1){
                            // c_R=sqrt(Y1_R*(gamma1*(p_R+pi1)/rho1_R)+(1.0-Y1_R)*(gamma2*(p_R+pi2)/rho2_R)); //frozen
                            // if (model==1) c_R=sqrt(1.0/(rho_R*(alpha1_R/(gamma1*(p_R+pi1))+alpha2_R/(gamma2*(p_R+pi2))))); //Wood
                            // else if (model==2) c_R=sqrt((Y1_R/(gamma1-1.0)*(gamma1*(p_R+pi1)/rho1_R)+(1.0-Y1_R)/(gamma2-1.0)*(gamma2*(p_R+pi2)/rho2_R))/(alpha1_R/(gamma1-1.0)+alpha2_R/(gamma2-1.0))); //Allaire
                            cc1_R=sound_speed_square(rho1_R,p_R,gamma1,pi1,eta1,cB11,cB21,cE11,cE21,rho01,e01);
                            cc2_R=sound_speed_square(rho2_R,p_R,gamma2,pi2,eta2,cB12,cB22,cE12,cE22,rho02,e02);
                            c_R=sqrt((Y1_R/(gamma1-1.0)*cc1_R+(1.0-Y1_R)/(gamma2-1.0)*cc2_R)/(alpha1_R/(gamma1-1.0)+alpha2_R/(gamma2-1.0)));
                            // c_R=sqrt((Y1_R/(gamma1-1.0)*(((Gamma1+1.0)*p_R-p_ref1)/rho1_R+(p_R-p_ref1)/Gamma1*dGamma1+dp_ref1-rho1_R*Gamma1*de_ref1)+(1.0-Y1_R)/(gamma2-1.0)*(((Gamma2+1.0)*p_R-p_ref2)/rho2_R+(p_R-p_ref2)/Gamma2*dGamma2+dp_ref2-rho2_R*Gamma2*de_ref2))/(alpha1_R/(gamma1-1.0)+alpha2_R/(gamma2-1.0)));
                        }
                        else if (sound_speed_type==2){
                            gamma_mix=1.0+1.0/(alpha1_R/(gamma1-1.0)+alpha2_R/(gamma2-1.0));
                            p_ref=(gamma_mix-1.0)/gamma_mix*(alpha1_R*pi1*gamma1/(gamma1-1.0)+alpha2_R*pi2*gamma2/(gamma2-1.0));
                            c_R=sqrt(gamma_mix*(p_R+p_ref)/rho_R);
                        }
                        // c_R=sqrt(alpha1_R*cc1_R+alpha2_R*cc2_R);
                        
                        // permutation of bases of velocity vector
                        velocity = {u_L, v_L, w_L};
                        u_L = velocity[id0];
                        v_L = velocity[id1];
                        w_L = velocity[id2];
                        velocity = {u_R, v_R, w_R};
                        u_R = velocity[id0];
                        v_R = velocity[id1];
                        w_R = velocity[id2];
                        
                        if (surface_tension_type>=1) kappa=curv[I_c(i,j,k)];
                        else kappa=0.0;
                        
                        // Direct Wave Speed Estimates
                        // S_L=min(u_L-c_L,u_R-c_R);
                        // S_R=max(u_L+c_L,u_R+c_R);
                        S_L=min({0.,u_L-c_L,(u_L+u_R)/2.-(c_L+c_R)/2.});
                        S_R=max({0.,u_R+c_R,(u_L+u_R)/2.+(c_L+c_R)/2.});
                        if (isnan(S_L) || isnan(S_R)) {
                            printf("SLR nan@xc"); getchar();
                        }
                        
                        // Maximum Wave Speed
                        // if (mwsx<max(fabs(S_L),fabs(S_R))){
                        //     mwsx=max(fabs(S_L),fabs(S_R));
                        // }

                        // contact discontinuity speed
                        S_star=((p_R-p_L)+(rho_L*u_L*(S_L-u_L)-rho_R*u_R*(S_R-u_R))-sigma_CSF*kappa*(alpha1_R-alpha1_L))
                                /
                                (rho_L*(S_L-u_L)-rho_R*(S_R-u_R));

                        // HLLC solution
                        U_L[0]=alpha1_L;
                        U_L[1]=alpha1_L*rho1_L;
                        U_L[2]=alpha2_L*rho2_L;
                        U_L[3]=rho_L*u_L;
                        U_L[4]=rho_L*v_L;
                        U_L[5]=rho_L*w_L;
                        U_L[6]=rhoE_L;
                        S_ratio_L=(S_L-u_L)/(S_L-S_star);
                        U_Lstar[0]=alpha1_L;//*S_ratio_L;
                        U_Lstar[1]=alpha1_L*rho1_L*S_ratio_L;
                        U_Lstar[2]=alpha2_L*rho2_L*S_ratio_L;
                        U_Lstar[3]=rho_L*S_star*S_ratio_L;
                        U_Lstar[4]=rho_L*v_L*S_ratio_L;
                        U_Lstar[5]=rho_L*w_L*S_ratio_L;
                        U_Lstar[6]=rho_L*S_ratio_L*(rhoE_L/rho_L+(S_star-u_L)*(S_star+(p_L-sigma_CSF*kappa*alpha1_L)/rho_L/(S_L-u_L)));
                        
                        U_R[0]=alpha1_R;
                        U_R[1]=alpha1_R*rho1_R;
                        U_R[2]=alpha2_R*rho2_R;
                        U_R[3]=rho_R*u_R;
                        U_R[4]=rho_R*v_R;
                        U_R[5]=rho_R*w_R;
                        U_R[6]=rhoE_R;
                        S_ratio_R=(S_R-u_R)/(S_R-S_star);
                        U_Rstar[0]=alpha1_R;//*S_ratio_R;
                        U_Rstar[1]=alpha1_R*rho1_R*S_ratio_R;
                        U_Rstar[2]=alpha2_R*rho2_R*S_ratio_R;
                        U_Rstar[3]=rho_R*S_star*S_ratio_R;
                        U_Rstar[4]=rho_R*v_R*S_ratio_R;
                        U_Rstar[5]=rho_R*w_R*S_ratio_R;
                        U_Rstar[6]=rho_R*S_ratio_R*(rhoE_R/rho_R+(S_star-u_R)*(S_star+(p_R-sigma_CSF*kappa*alpha1_R)/rho_R/(S_R-u_R)));
                        
                        // calculate fluctuation at cell inside
                        Ic = I_c(i,j,k); // index of cell
                        for (m=0;m<num_var;m++){
                            W1=U_Lstar[m]-U_L[m];
                            W2=U_Rstar[m]-U_Lstar[m];
                            W3=U_R[m]-U_Rstar[m];
                            Adq[m_list[m]][Ic]=(S_L*W1+S_R*W3)+S_star*W2;
                        }
                    }
                }
            }
        }
    }
    *MWS_d = mws_d;
}

void cal_dt(double MWS_x, double MWS_y, double MWS_z){
    
    // calculate dt
    if (dt_last > 0.0){
        dt = dt_last;
    }
    else {
        dt = CFL / (MWS_x / dx + MWS_y / dy + MWS_z / dz + eps);
    }
}

void update(int rk){
    int i, j, k, m, Ic, rk_next = (rk + 1) % RK_stage;
    
    #pragma omp parallel
    {
        int Ixm, Ixp, Iym, Iyp, Izm, Izp, s;
        double rho, rhou, rhoE, u, p;
        vec1d dqdt(num_var, 0.0); // spatial discrete operator
    
        #pragma omp for collapse(3) private(i, j, k, m, Ic)
        for (k = ngz; k < ngz + nz; k++){ // each cell in z-direction
            for (j = ngy; j < ngy + ny; j++){ // each cell in y-direction
                for (i = ngx; i < ngx + nx; i++){ // each cell in x-direction
                    Ic = I_c(i, j, k); // index of cell
                    Ixm = I_x(i, j, k); // index of cell boundary at x_{i-1/2}
                    Ixp = I_x(i + 1, j, k); // index of cell boundary at x_{i+1/2}
                    Iym = I_y(i, j, k); // index of cell boundary at y_{j-1/2}
                    Iyp = I_y(i, j + 1, k); // index of cell boundary at y_{j+1/2}
                    Izm = I_z(i, j, k); // index of cell boundary at z_{k-1/2}
                    Izp = I_z(i, j, k + 1); // index of cell boundary at z_{k+1/2}
                    
                    // calculate spatial discrete operator
                    if (spatial_discre == 1){
                        // divergence of flux tensor
                        for (m = 0; m < num_var; m++){
                            if      (dim == 1) dqdt[m] = -(F_x[m][Ixp] - F_x[m][Ixm]) / dx;
                            else if (dim == 2) dqdt[m] = -(F_x[m][Ixp] - F_x[m][Ixm]) / dx
                                                        -(F_y[m][Iyp] - F_y[m][Iym]) / dy;
                            else if (dim == 3) dqdt[m] = sum_cons3(-(F_x[m][Ixp] - F_x[m][Ixm]) / dx,
                                                                -(F_y[m][Iyp] - F_y[m][Iym]) / dy,
                                                                -(F_z[m][Izp] - F_z[m][Izm]) / dz);
                        }
                        // non-conservative terms in Allaire model
                        if      (dim == 1) dqdt[0] += U[rk][0][Ic]*((u_Riemann[Ixp]-u_Riemann[Ixm])/dx);
                        else if (dim == 2) dqdt[0] += U[rk][0][Ic]*((u_Riemann[Ixp]-u_Riemann[Ixm])/dx
                                                                    +(v_Riemann[Iyp]-v_Riemann[Iym])/dy);
                        else if (dim == 3) dqdt[0] += U[rk][0][Ic]*sum_cons3((u_Riemann[Ixp]-u_Riemann[Ixm])/dx,
                                                                            (v_Riemann[Iyp]-v_Riemann[Iym])/dy,
                                                                            (w_Riemann[Izp]-w_Riemann[Izm])/dz);
                        // non-conservative terms in Kapila model
                        // alpha1=U[k_RK][0][Ic]; alpha2=1.0-alpha1;
                        // cons_to_prim(&rho1,&rho2,&u,&v,&w,&p,
                        //     U[k_RK][0][Ic],U[k_RK][1][Ic],U[k_RK][2][Ic],U[k_RK][3][Ic],U[k_RK][4][Ic],U[k_RK][5][Ic],U[k_RK][6][Ic]);
                        // K=(gamma2*(p+pi2)-gamma1*(p+pi1))/(gamma1*(p+pi1)/alpha1+gamma2*(p+pi2)/alpha2);
                        // dqdt[0]+=(alpha1+K)*sum_cons3((u_HLLC[I_x(i+1,j,k)]-u_HLLC[I_x(i,j,k)])/dx,(v_HLLC[I_y(i,j+1,k)]-v_HLLC[I_y(i,j,k)])/dy,(w_HLLC[I_z(i,j,k+1)]-w_HLLC[I_z(i,j,k)])/dz);
                    }
                    else if (spatial_discre == 2){
                        for (m = 0; m < num_var; m++){
                            if (dim == 1) dqdt[m] = -((Amdq_x[m][Ixp] + Apdq_x[m][Ixm]) + Adq_x[m][Ic]) / dx;
                            if (dim == 2) dqdt[m] = -((Amdq_x[m][Ixp] + Apdq_x[m][Ixm]) + Adq_x[m][Ic]) / dx
                                                    -((Amdq_y[m][Iyp] + Apdq_y[m][Iym]) + Adq_y[m][Ic]) / dy;
                            if (dim == 3) dqdt[m] = sum_cons3(-((Amdq_x[m][Ixp] + Apdq_x[m][Ixm]) + Adq_x[m][Ic]) / dx,
                                                            -((Amdq_y[m][Iyp] + Apdq_y[m][Iym]) + Adq_y[m][Ic]) / dy,
                                                            -((Amdq_z[m][Izp] + Apdq_z[m][Izm]) + Adq_z[m][Ic]) / dz);
                        }
                    }
                    
                    // calculate surface tension term
                    if (surface_tension_type == 1 || surface_tension_type == 2){
                        F_CSF[0][Ic]=sigma_CSF*curv[Ic]*(VOF_x_Riemann[Ixp]-VOF_x_Riemann[Ixm])/dx;
                        F_CSF[1][Ic]=sigma_CSF*curv[Ic]*(VOF_y_Riemann[Iyp]-VOF_y_Riemann[Iym])/dy;
                        if (dim >= 1) dqdt[3]+=sigma_CSF*curv[Ic]*(VOF_x_Riemann[Ixp]-VOF_x_Riemann[Ixm])/dx;
                        if (dim >= 2) dqdt[4]+=sigma_CSF*curv[Ic]*(VOF_y_Riemann[Iyp]-VOF_y_Riemann[Iym])/dy;
                        if (dim >= 3) dqdt[5]+=sigma_CSF*curv[Ic]*(VOF_z_Riemann[Izp]-VOF_z_Riemann[Izm])/dz;
                        
                        if (dim == 1){
                            dqdt[6]+=sigma_CSF*curv[Ic]*((VOF_x_Riemann[Ixp]*u_Riemann[Ixp]-VOF_x_Riemann[Ixm]*u_Riemann[Ixm])/dx
                                                        -U[rk][0][Ic]*(u_Riemann[Ixp]-u_Riemann[Ixm])/dx);
                        }
                        else if (dim == 2){
                            dqdt[6]+=sigma_CSF*curv[Ic]*(((VOF_x_Riemann[Ixp]*u_Riemann[Ixp]-VOF_x_Riemann[Ixm]*u_Riemann[Ixm])/dx
                                                        +(VOF_y_Riemann[Iyp]*v_Riemann[Iyp]-VOF_y_Riemann[Iym]*v_Riemann[Iym])/dy)
                                                        -U[rk][0][Ic]*((u_Riemann[Ixp]-u_Riemann[Ixm])/dx
                                                                      +(v_Riemann[Iyp]-v_Riemann[Iym])/dy));
                        }
                        else if (dim == 3){
                            dqdt[6]+=sigma_CSF*curv[Ic]*(sum_cons3((VOF_x_Riemann[Ixp]*u_Riemann[Ixp]-VOF_x_Riemann[Ixm]*u_Riemann[Ixm])/dx,
                                                                   (VOF_y_Riemann[Iyp]*v_Riemann[Iyp]-VOF_y_Riemann[Iym]*v_Riemann[Iym])/dy,
                                                                   (VOF_z_Riemann[Izp]*w_Riemann[Izp]-VOF_z_Riemann[Izm]*w_Riemann[Izm])/dz)
                                    -U[rk][0][Ic]*sum_cons3((u_Riemann[Ixp]-u_Riemann[Ixm])/dx,
                                                            (v_Riemann[Iyp]-v_Riemann[Iym])/dy,
                                                            (w_Riemann[Izp]-w_Riemann[Izm])/dz));
                        }
                    }
                    
                    // obtain numerical solution at next sub-step in Runge Kutta method
                    for (m = 0; m < num_var; m++){
                        U[rk_next][m][Ic] = RK_alpha[rk][0] * U[0][m][Ic];
                        for (s = 1; s <= rk; s++){
                            U[rk_next][m][Ic] += RK_alpha[rk][s] * U[s][m][Ic];
                        }
                        U[rk_next][m][Ic] += RK_beta[rk] * dqdt[m] * dt;
                    }
                    
                    // check positivity
                    // rho = U[rk_next][0][Ic];
                    // rhou = U[rk_next][1][Ic];
                    // rhoE = U[rk_next][2][Ic];
                    // u = rhou / rho;
                    // p = (gamma_ - 1.0) * (rhoE - 0.5 * rho * u * u);
                    // if (rho <= 0.0 || p <= 0.0){
                    //     cout << "rho or p < 0" << endl;
                    //     cout << "rho = " << rho << ", p = " << p << endl;
                    //     getchar();
                    // }
                }
            }
        }
    }
}

void output_result(string file_name){
    int i, j, k, m, Ic, d, rk;
    double alpha1, alpha1rho1, alpha2rho2, rhou, rhov, rhow, rhoE, rho1, rho2, u, v, w, p, rho, Y1;
    int BVD_func;
    int Ixm, Ixp, Iym, Iyp, Izm, Izp;
    
    std::ofstream file_result(file_name);
    
    // file_result << "#x,y,z,alpha1,rho1,rho2,u,v,w,p,rho,Y1,BVD_func\n";
    file_result << "x,y,z,alpha1,rho1,rho2,u,v,w,p,rho,Y1,kappa,Fx,Fy,gradColx,gradColy,uRiemann,vRiemann,BVD_func\n";
    for (i = ngx; i < ngx + nx; i++){
        for (j = ngy; j < ngy + ny; j++){
            for (k = ngz; k < ngz + nz; k++){
                // index of cell
                Ic = I_c(i, j, k);
                Ixm = I_x(i, j, k); // index of cell boundary at x_{i-1/2}
                Ixp = I_x(i + 1, j, k); // index of cell boundary at x_{i+1/2}
                Iym = I_y(i, j, k); // index of cell boundary at y_{j-1/2}
                Iyp = I_y(i, j + 1, k); // index of cell boundary at y_{j+1/2}
                Izm = I_z(i, j, k); // index of cell boundary at z_{k-1/2}
                Izp = I_z(i, j, k + 1); // index of cell boundary at z_{k+1/2}
                
                alpha1 = U[0][0][Ic];
                alpha1rho1 = U[0][1][Ic];
                alpha2rho2 = U[0][2][Ic];
                rhou = U[0][3][Ic];
                rhov = U[0][4][Ic];
                rhow = U[0][5][Ic];
                rhoE = U[0][6][Ic];
                
                cons_to_prim_5eq(&rho1,&rho2,&u,&v,&w,&p,
                                alpha1,alpha1rho1,alpha2rho2,rhou,rhov,rhow,rhoE);
                
                rho = alpha1 * rho1 + (1.0 - alpha1) * rho2;
                Y1 = alpha1 * rho1 / rho;
                
                file_result << std::setprecision(15);
                file_result << xc[i] << "," << yc[j] << "," << zc[k] << ",";
                file_result
                    << alpha1 << ","
                    << rho1 << ","
                    << rho2 << ","
                    << u << ","
                    << v << ","
                    << w << ","
                    << p << ","
                    << rho << ","
                    << Y1 << ","
                    << curv[Ic] << ","
                    << F_CSF[0][Ic] << ","
                    << F_CSF[1][Ic] << ","
                    << 0.5*(u_Riemann[Ixm]+u_Riemann[Ixp]) << ","
                    << 0.5*(v_Riemann[Iym]+v_Riemann[Iyp]) << ","
                    << grad_VOF[0][Ic] << ","
                    << grad_VOF[1][Ic] << ","
                    ;
                BVD_func = 0;
                for (d = 0; d < dim; d++){
                    for (rk = 0; rk < RK_stage; rk++){
                        for (m = 0; m < num_var; m++){
                            BVD_func = fmax(BVD_func, BVD_active[d][rk][m][Ic]);
                        }
                    }
                }
                file_result << BVD_func << "\n";
                // file_result << "\n";
            }
        }
    }
    
    file_result.close();
}

void plot_result(){
    
    FILE *gp;
    string variable_name;
    double yr_min, yr_max;
    
    int plot_var = 1; // 1: volume fraction
    if      (plot_var == 1) variable_name = "volume fraction";
    // else if (plot_var == 2) variable_name = "velocity";
    // else if (plot_var == 3) variable_name = "pressure";
    // else if (plot_var == 4) variable_name = "BVD_active";
    
    // if (problem_type == 1){ // Sod problem
    //     yr_min = 0.0; yr_max = 1.2;
    // }
    // else if (problem_type == 2){ // Le Blanc problem
    //     yr_min = 1.0e-3; yr_max = 1.0;
    // }
    yr_min = 0.0; yr_max = 1.2;
    
    #if defined(_WIN32) || defined(_WIN64)
        // for windows pc
        gp = _popen(GNUPLOT, "w");
        if (!gp){
            cout << "unable to start gnuplot" << endl;
        }
        else {
            // plot numerical result using gnuplot
            fprintf(gp, "set term wxt 1\n");
            fprintf(gp, "set size ratio 1\n");
            fprintf(gp, "set yr[%f:%f]\n", yr_min, yr_max);
            fprintf(gp, "set xl \"x\"\n");
            fprintf(gp, "set yl \"%s\"\n", variable_name.c_str());
            fprintf(gp, "set title \"1D Euler\"\n");
            fprintf(gp, "plot \"./result.csv\" using %d:%d title \"%s\" w lp lt 7 ps 1\n", 1, plot_var + 1, scheme_name.c_str());
            fflush(gp);
            _pclose(gp);
        }
        
    #elif defined(__APPLE__) && defined(__MACH__)
        // for mac pc
        gp = popen(GNUPLOT, "w");
        if (!gp){
            cout << "unable to start gnuplot" << endl;
        }
        else {
            // plot numerical result using gnuplot
            // fprintf(gp, "set term wxt 1\n");
            fprintf(gp, "set size ratio 1\n");
            fprintf(gp, "set yr[%f:%f]\n", yr_min, yr_max);
            fprintf(gp, "set xl \"x\"\n");
            fprintf(gp, "set yl \"%s\"\n", variable_name.c_str());
            fprintf(gp, "set title \"1D Euler\"\n");
            fprintf(gp, "plot \"./result.csv\" using %d:%d title \"%s\" w lp lt 7 ps 1\n", 1, plot_var + 1, scheme_name.c_str());
            fflush(gp);
            pclose(gp);
        }
        
    #elif defined(__linux__)
        // for linux pc
        gp = popen(GNUPLOT, "w");
        if (!gp){
            cout << "unable to start gnuplot" << endl;
        }
        else {
            // plot numerical result using gnuplot
            fprintf(gp, "set term wxt 1\n");
            fprintf(gp, "set size ratio 1\n");
            // fprintf(gp, "set logscale y\n");
            fprintf(gp, "set yr[%f:%f]\n", yr_min, yr_max);
            fprintf(gp, "set xl \"x\"\n");
            fprintf(gp, "set yl \"%s\"\n", variable_name.c_str());
            fprintf(gp, "set title \"1D Euler\"\n");
            fprintf(gp, "plot \"./result.csv\" using %d:%d title \"%s\" w lp lt 7 ps 1\n", 1, plot_var + 3, scheme_name.c_str());
            fflush(gp);
            pclose(gp);
        }
        
    #endif
        
}

double sign(double a){
    if (a > 0.0)      return 1.0;
    else if (a < 0.0) return -1.0;
    else              return 0.0;
}

inline int I_c(int i, int j, int k){
    return NX * NY * k + NX * j + i;
}

inline int I_x(int i, int j, int k){
    return (NX + 1) * NY * k + (NX + 1) * j + i;
}

inline int I_y(int i, int j, int k){
    return NX * (NY + 1) * k + NX * j + i;
}

inline int I_z(int i, int j, int k){
    return NX * NY * k + NX * j + i;
}

inline void get_cb_indices_from_cc(int d, int i, int j, int k, int& Im, int& Ip){
    switch (d) {
        case 0:
            Im = I_x(i, j, k);
            Ip = I_x(i + 1, j, k);
            break;
        case 1:
            Im = I_y(i, j, k);
            Ip = I_y(i, j + 1, k);
            break;
        case 2:
            Im = I_z(i, j, k);
            Ip = I_z(i, j, k + 1);
            break;
    }
}


double q2(double u,double v,double w){
    return sum_cons3(u*u,v*v,w*w);
}

double sum_cons3(double a, double b, double c){
    double sum;
    if (SYMP == 1){
        double s1,s2,s3;
        s1=(a+b)+c;
        s2=(c+a)+b;
        s3=(b+c)+a;
        sum=0.5*(std::min({s1,s2,s3})+std::max({s1,s2,s3}));
    }
    else {
        sum=a+b+c;
    }
    return sum;
}

double pow_int(double x,int n){
    double xn=x;
    for (int i=0;i<n-1;i++){
        xn=xn*x;
    }
    return xn;
}

void CSFmodel(int rk){
    if (dim >= 1) grad_VOF_cal_dim(0, rk); // calculate gradient in x-direction
    if (dim >= 2) grad_VOF_cal_dim(1, rk); // calculate gradient in y-direction
    if (dim >= 3) grad_VOF_cal_dim(2, rk); // calculate gradient in z-direction
    curv_cal();
    
    if (surface_tension_type == 2) curv_filter(rk); // apply curvature filter
}

void grad_VOF_cal_dim(int d, int rk){
    int i, j, k, xi, Ic;
    int nsc = order_grad_VOF + 1;
    int smoothing = 1;
    double pow_sm = 0.1;
    vec1d ddx = {dx, dy, dz};
    
    #pragma omp parallel
    {
        vec1d stencil(nsc); // stencil for gradient calculation
        vec1i idx(3);
        
        #pragma omp for collapse(3) private(k, j, i, Ic, xi)
        for (k = ngz; k < ngz + nz; k++){ // each cell in z-direction
            for (j = ngy; j < ngy + ny; j++){ // each cell in y-direction
                for (i = ngx; i < ngx + nx; i++){ // each cell in x-direction
                    Ic = I_c(i, j, k);
                    idx = {i, j, k};
                    idx[d] = idx[d] - (nsc - 1) / 2;
                    
                    for (xi = 0; xi < nsc; xi++){
                        stencil[xi] = U[rk][0][I_c(idx[0], idx[1], idx[2])]; // calculate stencil centered at the cell
                        idx[d]++; // increment index for next stencil point
                        // in the case of smoothing
                        if (smoothing) stencil[xi] = pow(stencil[xi], pow_sm) / (pow(stencil[xi], pow_sm) + pow(1.0 - stencil[xi], pow_sm));
                    }
                    
                    // calculate gradient value at cell center
                    grad_VOF[d][Ic] = 0.0;
                    for (xi = 0; xi < (nsc - 1) / 2; xi++){
                        grad_VOF[d][Ic] += (coef_Pn_D1_CC[xi] * stencil[xi] + coef_Pn_D1_CC[nsc - 1 - xi] * stencil[nsc - 1 - xi]);
                    }
                    grad_VOF[d][Ic] += coef_Pn_D1_CC[(nsc - 1) / 2] * stencil[(nsc - 1) / 2];
                    grad_VOF[d][Ic] /= ddx[d];
                }
            }
        }
    }
}

void curv_cal(){
    int i,j,k,Ic;
    double alpha1,eps_int=1.0e-4;
    
    // Exact curvature in static droplet2
    if (0){
        // for (i=0;i<nx;i++){
        //     for (j=0;j<ny;j++){
        //         for (k=0;k<nz;k++){
        //             Ic=I(ng_CSF+i,ng_CSF+j,ng_CSF+k);
        //             alpha1=U[k_RK][0][Ic];
        //             if (((eps_int<alpha1) && (alpha1<1.0-eps_int)) && 1){
        //                 curv[Ic]=1.0/max(sqrt(sum_cons3(pow2(xc[i]),pow2(yc[j]),pow2(zc[k]))),eps);
        //                 // curv[Ic]=1.0;
        //             }
        //             else {
        //                 curv[Ic]=0.0;
        //             }
        //         }
        //     }
        // }
    }
    
    // Linear interpolation
    #pragma omp parallel
    {
        // int ii,jj;
        // double grad_VOF_x,grad_VOF_y;
        int xi,kc;
        int nsc = 3;
        double norm,dnxdx,dnydy,dnzdz;
        bool normal_exist;
        
        #pragma omp for collapse(3) private(i,j,k,Ic)
        for (k = ngz; k < ngz + nz; k++){ // each cell in z-direction
            for (j = ngy; j < ngy + ny; j++){ // each cell in y-direction
                for (i = ngx; i < ngx + nx; i++){ // each cell in x-direction
                    Ic = I_c(i, j, k);
                    curv[Ic] = 0.0;
                    
                    // normalization of gradient vector
                    if      (dim == 1) norm = fabs(grad_VOF[0][Ic]);
                    else if (dim == 2) norm = sqrt(pow_int(grad_VOF[0][Ic], 2) + pow_int(grad_VOF[1][Ic], 2));
                    else if (dim == 3) norm = sqrt(sum_cons3(pow_int(grad_VOF[0][Ic], 2), pow_int(grad_VOF[1][Ic], 2), pow_int(grad_VOF[2][Ic], 2)));
                    if (norm>1.0e-20 || 0){
                        normal_vec[0][Ic]=grad_VOF[0][Ic]/norm;
                        normal_vec[1][Ic]=grad_VOF[1][Ic]/norm;
                        normal_vec[2][Ic]=grad_VOF[2][Ic]/norm;
                        normal_vec[3][Ic]=1.0;
                    }
                    else {
                        normal_vec[0][Ic]=normal_vec[1][Ic]=normal_vec[2][Ic]=0.0;
                        normal_vec[3][Ic]=0.0;
                    }
                }
            }
        }
        #pragma omp for collapse(3) private(i,j,k,Ic)
        for (k = ngz; k < ngz + nz; k++){ // each cell in z-direction
            for (j = ngy; j < ngy + ny; j++){ // each cell in y-direction
                for (i = ngx; i < ngx + nx; i++){ // each cell in x-direction
                    Ic = I_c(i, j, k);
                    normal_exist=normal_vec[3][Ic]==1.0;
                    if (dim>=1) normal_exist=normal_exist && (normal_vec[3][I_c(i+1,j,k)]==1.0) && (normal_vec[3][I_c(i-1,j,k)]==1.0);
                    if (dim>=2) normal_exist=normal_exist && (normal_vec[3][I_c(i,j+1,k)]==1.0) && (normal_vec[3][I_c(i,j-1,k)]==1.0);
                    if (dim>=3) normal_exist=normal_exist && (normal_vec[3][I_c(i,j,k+1)]==1.0) && (normal_vec[3][I_c(i,j,k-1)]==1.0);
                    
                    dnxdx=dnydy=dnzdz=0.0;
                    if (normal_exist || 0){
                        if (dim >= 1) dnxdx=(normal_vec[0][I_c(i+1,j,k)]-normal_vec[0][I_c(i-1,j,k)])/(2.0*dx);
                        if (dim >= 2) dnydy=(normal_vec[1][I_c(i,j+1,k)]-normal_vec[1][I_c(i,j-1,k)])/(2.0*dy);
                        if (dim >= 3) dnzdz=(normal_vec[2][I_c(i,j,k+1)]-normal_vec[2][I_c(i,j,k-1)])/(2.0*dz);
                    }
                    // else {
                    //     dnxdx=dnydy=dnzdz=0.0;
                    // }
                    
                    // calculate curvature
                    curv[Ic]=-sum_cons3(dnxdx,dnydy,dnzdz);
                    // curv[Ic]=1.0; // set curvature to 1.0 for testing
                }
            }
        }
    }
}

void curv_filter(int rk){
    int i, j, k, Ic;
    double alpha1;
    
    // filtering strategy
    #pragma omp parallel
    {
        // int ii,jj;
        
        #pragma omp for collapse(3) private(i,j,k,Ic,alpha1)
        for (k = ngz; k < ngz + nz; k++){ // each cell in z-direction
            for (j = ngy; j < ngy + ny; j++){ // each cell in y-direction
                for (i = ngx; i < ngx + nx; i++){ // each cell in x-direction
                    Ic = I_c(i, j, k);
                    alpha1=U[rk][0][Ic];
                    weight_filter[Ic]=pow_int(alpha1*(1.0-alpha1),2);
                }
            }
        }
        #pragma omp for collapse(3) private(i,j,k,Ic)
        for (k = ngz; k < ngz + nz; k++){ // each cell in z-direction
            for (j = ngy + 1; j < ngy + ny - 1; j++){ // each cell in y-direction
                for (i = ngx + 1; i < ngx + nx - 1; i++){ // each cell in x-direction
                    Ic = I_c(i, j, k);
                    weight_sum_filter[Ic]=weight_filter[Ic]
                        +((weight_filter[I_c(i-1,j-1,k)]+weight_filter[I_c(i+1,j+1,k)])
                        + (weight_filter[I_c(i+1,j-1,k)]+weight_filter[I_c(i-1,j+1,k)]))
                        +((weight_filter[I_c(i,j-1,k)]+weight_filter[I_c(i,j+1,k)])
                        + (weight_filter[I_c(i-1,j,k)]+weight_filter[I_c(i+1,j,k)]));
                    weight_sum_filter[Ic]=max(weight_sum_filter[Ic],1.0e-20);
                }
            }
        }
    }
    int n_itr = 0, N_itr = 5;
    while (n_itr < N_itr){
        #pragma omp parallel
        {
            #pragma omp for collapse(3) private(i,j,k,Ic)
            for (k=ngz;k<ngz+nz;k++){ // each cell in z-direction
                for (j=ngy+1;j<ngy+ny-1;j++){ // each cell in y-direction
                    for (i=ngx+1;i<ngx+nx-1;i++){ // each cell in x-direction
                        Ic=I_c(i,j,k);
                        curv_sum_filter[Ic]=weight_filter[Ic]*curv[Ic]
                            +((weight_filter[I_c(i-1,j-1,k)]*curv[I_c(i-1,j-1,k)]+weight_filter[I_c(i+1,j+1,k)]*curv[I_c(i+1,j+1,k)])
                            + (weight_filter[I_c(i+1,j-1,k)]*curv[I_c(i+1,j-1,k)]+weight_filter[I_c(i-1,j+1,k)]*curv[I_c(i-1,j+1,k)]))
                            +((weight_filter[I_c(i,j-1,k)]*curv[I_c(i,j-1,k)]+weight_filter[I_c(i,j+1,k)]*curv[I_c(i,j+1,k)])
                            + (weight_filter[I_c(i-1,j,k)]*curv[I_c(i-1,j,k)]+weight_filter[I_c(i+1,j,k)]*curv[I_c(i+1,j,k)]));
                    }
                }
            }
            #pragma omp for collapse(3) private(i,j,k,Ic)
            for (k=ngz;k<ngz+nz;k++){ // each cell in z-direction
                for (j=ngy+1;j<ngy+ny-1;j++){ // each cell in y-direction
                    for (i=ngx+1;i<ngx+nx-1;i++){ // each cell in x-direction
                        Ic=I_c(i,j,k);
                        curv[Ic]=curv_sum_filter[Ic]/weight_sum_filter[Ic];
                        if (isnan(curv[Ic])){
                            printf("stop\n"); getchar();
                        }
                    }
                }
            }
        }
        n_itr++;
    }
}

void coefficient_linear_polynoimal_1stDerivative_CellCenter(int n){
    
    double deno;
    
    if (n==3){
        coef_Pn_D1_CC={-1., 0., 1.};
        deno=2.;
    }
    else if (n==5){
        coef_Pn_D1_CC={5., -34., 0., 34., -5.};
        deno=48.;
    }
    else if (n==7){
        coef_Pn_D1_CC={-259., 2236., -9455., 0., 9455., -2236., 259.};
        deno=11520.;
    }
    else if (n==9){
        coef_Pn_D1_CC={3229., -33878., 170422., -574686., 0., 574686., -170422., 33878., -3229.};
        deno=645120.;
    }
    else if (n==11){
        coef_Pn_D1_CC={-117469., 1456392., -8592143., 32906032., -96883458., 0., 96883458., -32906032., 8592143., -1456392., 117469.};
        deno=103219200.;
    }
    else if (n==13){
        coef_Pn_D1_CC={7156487., -102576686., 699372916., -3055539322., 9868012803., -26521889196., 0., 26521889196., -9868012803., 3055539322., -699372916., 102576686., -7156487.};
        deno=27249868800.;
    }
    else if (n==15){
        coef_Pn_D1_CC={-2430898831., 39590631044., -307360078831., 1523913922544., -5491720851331., 15758300772500., -39658726267875., 0., 39658726267875., -15758300772500., 5491720851331., -1523913922544., 307360078831., -39590631044., 2430898831.};
        deno=39675808972800.;
    }
    else if (n==17){
        coef_Pn_D1_CC={4574844075., -83495007698., 728461015102., -4060076056898., 16354419488602., -51427361405498., 135225244018150., -323811837170250., 0., 323811837170250., -135225244018150., 51427361405498., -16354419488602., 4060076056898., -728461015102., 83495007698., -4574844075.};
        deno=317406471782400.;
    }
    
    for (int k=0;k<n;k++){
        coef_Pn_D1_CC[k]/=deno;
    }
    
}