#ifndef CUDAFUNCTIONS_H_
#define CUDAFUNCTIONS_H_

// declare CUDA fields
T *dT , *dr , *dp , *dq , *ds;
T *dcc, *dff, *dss, *dww;  // matrix A center, south, west stencil
//  --------------
T *kww, *kw,  *kc;  // (TNS2) matrix M-1 with 12+1 point stencil
T       *ksw, *ks, *kse;
T             *kss;
//  --------------
T       *kfn;
T *kfw, *kf, *kfe;
T       *kfs;
//  --------------
T       *kff;
//  --------------
T *dpp;              // matrix P = sqrt(D)
T *drh, *dsg;        // partial dot products
T *dV;               // cell volume
T *dqB;              // Neumann BC bottom



// initialize CUDA fields
template <class T>
void cudaInit( T *hT,
		T *hV,
		T *hcc,
		T *hff,
		T *hss,
		T *hww,
		T *hqB,
		const int blocks,
		const int Nx,
		const int Ny,
		const int Nz)
{
	cudaMalloc((void**)&dT ,sizeof(T)*(Nx*Ny*Nz+2*Nx*Ny));
	cudaMalloc((void**)&dr ,sizeof(T)*(Nx*Ny*Nz+2*Nx*Ny));
	cudaMalloc((void**)&dV ,sizeof(T)*(Nx*Ny*Nz+2*Nx*Ny));
	cudaMalloc((void**)&dp ,sizeof(T)*(Nx*Ny*Nz+2*Nx*Ny));
	cudaMalloc((void**)&dq ,sizeof(T)*(Nx*Ny*Nz+2*Nx*Ny));
	cudaMalloc((void**)&ds ,sizeof(T)*(Nx*Ny*Nz+2*Nx*Ny));
	cudaMalloc((void**)&dpp,sizeof(T)*(Nx*Ny*Nz+2*Nx*Ny));
	cudaMalloc((void**)&dcc,sizeof(T)*(Nx*Ny*Nz+2*Nx*Ny));
	cudaMalloc((void**)&dff,sizeof(T)*(Nx*Ny*Nz+Nx*Ny  ));
	cudaMalloc((void**)&dss,sizeof(T)*(Nx*Ny*Nz+Nx     ));
	cudaMalloc((void**)&dww,sizeof(T)*(Nx*Ny*Nz+1      ));
	cudaMalloc((void**)&drh,sizeof(T)*(blocks          ));
	cudaMalloc((void**)&dsg,sizeof(T)*(blocks          ));
	cudaMalloc((void**)&dqB,sizeof(T)*(Nx*Ny           ));


	cudaMalloc((void**)&kc ,sizeof(T)*(Nx*Ny*Nz+2*Nx*Ny )); // 12+1 diagonals for M^-1
	cudaMalloc((void**)&kff,sizeof(T)*(Nx*Ny*Nz+2*Nx*Ny ));
	cudaMalloc((void**)&kfs,sizeof(T)*(Nx*Ny*Nz+Nx*Ny+Nx));
	cudaMalloc((void**)&kfw,sizeof(T)*(Nx*Ny*Nz+Nx*Ny+1 ));
	cudaMalloc((void**)&kf ,sizeof(T)*(Nx*Ny*Nz+Nx*Ny   ));
	cudaMalloc((void**)&kfe,sizeof(T)*(Nx*Ny*Nz+Nx*Ny-1 ));
	cudaMalloc((void**)&kfn,sizeof(T)*(Nx*Ny*Nz+Nx*Ny-Nx));
	cudaMalloc((void**)&kss,sizeof(T)*(Nx*Ny*Nz+2*Nx    ));
	cudaMalloc((void**)&ksw,sizeof(T)*(Nx*Ny*Nz+Nx+1    ));
	cudaMalloc((void**)&ks ,sizeof(T)*(Nx*Ny*Nz+Nx      ));
	cudaMalloc((void**)&kse,sizeof(T)*(Nx*Ny*Nz+Nx-1    ));
	cudaMalloc((void**)&kww,sizeof(T)*(Nx*Ny*Nz+2       ));
	cudaMalloc((void**)&kw ,sizeof(T)*(Nx*Ny*Nz+1       ));


	cudaMemcpy(dT ,hT ,sizeof(T)*(Nx*Ny*Nz+2*Nx*Ny),cudaMemcpyHostToDevice);
	cudaMemcpy(dcc,hcc,sizeof(T)*(Nx*Ny*Nz+2*Nx*Ny),cudaMemcpyHostToDevice);
	cudaMemcpy(dV, hV ,sizeof(T)*(Nx*Ny*Nz+2*Nx*Ny),cudaMemcpyHostToDevice);
	cudaMemcpy(dff,hff,sizeof(T)*(Nx*Ny*Nz+Nx*Ny  ),cudaMemcpyHostToDevice);
	cudaMemcpy(dss,hss,sizeof(T)*(Nx*Ny*Nz+Nx     ),cudaMemcpyHostToDevice);
	cudaMemcpy(dww,hww,sizeof(T)*(Nx*Ny*Nz+1      ),cudaMemcpyHostToDevice);
	cudaMemcpy(dqB,hqB,sizeof(T)*(Nx*Ny           ),cudaMemcpyHostToDevice);

	cudaMemset(dr ,0,sizeof(T)*(Nx*Ny*Nz+2*Nx*Ny));
	cudaMemset(dp ,0,sizeof(T)*(Nx*Ny*Nz+2*Nx*Ny));
	cudaMemset(dq ,0,sizeof(T)*(Nx*Ny*Nz+2*Nx*Ny));
	cudaMemset(ds ,0,sizeof(T)*(Nx*Ny*Nz+2*Nx*Ny));
	cudaMemset(dpp,0,sizeof(T)*(Nx*Ny*Nz+2*Nx*Ny));
	cudaMemset(drh,0,sizeof(T)*(blocks          ));
	cudaMemset(dsg,0,sizeof(T)*(blocks          ));

	cudaMemset(kc ,0, sizeof(T)*(Nx*Ny*Nz+2*Nx*Ny ));
	cudaMemset(kff,0, sizeof(T)*(Nx*Ny*Nz+2*Nx*Ny ));
	cudaMemset(kfs,0, sizeof(T)*(Nx*Ny*Nz+Nx*Ny+Nx));
	cudaMemset(kfw,0, sizeof(T)*(Nx*Ny*Nz+Nx*Ny+1 ));
	cudaMemset(kf ,0, sizeof(T)*(Nx*Ny*Nz+Nx*Ny   ));
	cudaMemset(kfe,0, sizeof(T)*(Nx*Ny*Nz+Nx*Ny-1 ));
	cudaMemset(kfn,0, sizeof(T)*(Nx*Ny*Nz+Nx*Ny-Nx));
	cudaMemset(kss,0, sizeof(T)*(Nx*Ny*Nz+2*Nx    ));
	cudaMemset(ksw,0, sizeof(T)*(Nx*Ny*Nz+Nx+1    ));
	cudaMemset(ks ,0, sizeof(T)*(Nx*Ny*Nz+Nx      ));
	cudaMemset(kse,0, sizeof(T)*(Nx*Ny*Nz+Nx-1    ));
	cudaMemset(kww,0, sizeof(T)*(Nx*Ny*Nz+2       ));
	cudaMemset(kw ,0, sizeof(T)*(Nx*Ny*Nz+1       ));
}



// destroy CUDA fields
void cudaFinalize()
{
	cudaFree(dT);
	cudaFree(dr);
	cudaFree(dp);
	cudaFree(dq);
	cudaFree(ds);
	cudaFree(dV);
	cudaFree(dpp);
	cudaFree(dcc);
	cudaFree(dff);
	cudaFree(dss);
	cudaFree(dww);
	cudaFree(drh);
	cudaFree(dsg);
	cudaFree(dqB);

	cudaFree(kc);  // TNS2
	cudaFree(kff);
	cudaFree(kfs);
	cudaFree(kfw);
	cudaFree(kf);
	cudaFree(kfe);
	cudaFree(kfn);
	cudaFree(kss);
	cudaFree(ksw);
	cudaFree(ks);
	cudaFree(kse);
	cudaFree(kww);
	cudaFree(kw);

	cudaDeviceReset();
}

// AXPY (y := alpha*x + beta*y)
template <class T>
__global__ void AXPY(T *y,
		const T *x,
		const T alpha,
		const T beta,
		const int Nx,
		const int Ny)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x + Nx*Ny;
	y[tid] = alpha * x[tid] + beta * y[tid];
}

// SPMV (sparse matrix-vector multiplication)
template <class T>
__global__ void SpMVv1(T *y,
		const T *stC,
		const T *stF,
		const T *stS,
		const T *stW,
		const T *x,
		const int Nx,
		const int Ny)
{
	int tid  = threadIdx.x + blockIdx.x * blockDim.x;
	int tids = tid + Nx*Ny;  // tid shifted

	y[tids] = stC[tid+Nx*Ny] * x[tids]        // center
	    	+ stS[tid+Nx]    * x[tids+Nx]     // north               N
	        + stW[tid+1]     * x[tids+1]      // east              W C E
	        + stS[tid]       * x[tids-Nx]     // south               S
	        + stW[tid]       * x[tids-1]      // west
	        + stF[tid+Nx*Ny] * x[tids+Nx*Ny]  // back                B
	        + stF[tid]       * x[tids-Nx*Ny]; // front               F
}


// SPMV (sparse matrix-vector multiplication)
template <class T>
__global__ void SpMVv2(T *y,
		const T *stC,
		const T *stFF,
		const T *stFS,
		const T *stFW,
		const T *stF,
		const T *stFE,
		const T *stFN,
		const T *stSS,
		const T *stSW,
		const T *stS,
		const T *stSE,
		const T *stWW,										//						24+1 diagonals
		const T *stW,
		const T *x,          								//                          BB
		const int Nx, 										//           ---------------------------------
		const int Ny,										//                          BN
		const int NxNyNz,									//                          |
		const int NxNy)    									//                      BW- B - BE
	       	                    							//                          |
{                                                			//                          BS
	int tid  = threadIdx.x + blockIdx.x * blockDim.x;		//       ---------------------------------
	int tids = tid + NxNy;  	                    		// tid shifted              NN
                                                 			//                          |
	y[tids] = stC[tid+NxNy]      * x[tids]         			// center               NW- N - NE
		    + stS[tid+Nx]        * x[tids+Nx]      			// north                    |
		    + stW[tid+1]         * x[tids+1]       			// east            WW - W - C - E - EE
		    + stS[tid]           * x[tids-Nx]      			// south                    |
		    + stW[tid]           * x[tids-1]       			// west                 SW- S - SE
		    + stF[tid+NxNy]      * x[tids+NxNy]   			// back                     |
		    + stF[tid]           * x[tids-NxNy]   			// front                    SS
		    + stSE[tid]          * x[tids-Nx+1]    			// south-east --------------------------------
		    + stSE[tid+Nx-1]     * x[tids+Nx-1]    			// north-west               FN
		    + stFE[tid]          * x[tids-NxNy+1] 			// front-east               |
		    + stFE[tid+NxNy-1]   * x[tids+NxNy-1] 			// back-west            FW- F -FE

            + stSW[tid]          * x[tids-Nx-1]     		// south-west               |
            + stSW[tid+Nx+1]     * x[tids+Nx+1]     		// north-east               FS
            + stSS[tid]          * x[tids-2*Nx]     		// south-south   ------------------------------
            + stSS[tid+2*Nx]     * x[tids+2*Nx]     		// north-north              FF
	        + stWW[tid]          * x[tids-2]        		// west-west
            + stWW[tid+2]        * x[tids+2]        		// east-east
            + stFN[tid]          * x[tids-NxNy+Nx]  		// front-north
            + stFN[tid+NxNy-Nx]  * x[tids+NxNy-Nx]; 		// back-south


	if (tid>0)					y[tids] += stFW[tid]         * x[tids-NxNy-1 ];		// front-west
	if (tid<NxNyNz-1)		    y[tids] += stFW[tid+NxNy+1]  * x[tids+NxNy+1 ];		// back-east
	if (tid>Nx-1)				y[tids] += stFS[tid]         * x[tids-NxNy-Nx];		// front-south
	if (tid<NxNyNz-Nx)		    y[tids] += stFS[tid+NxNy+Nx] * x[tids+NxNy+Nx];		// back-north
	if (tid>NxNy-1)   			y[tids] += stFF[tid]		 * x[tids-2*NxNy ];		// front-front
    if (tid<NxNyNz-NxNy)		y[tids] += stFF[tid+2*NxNy]  * x[tids+2*NxNy ];		// back-back
}

// DOT PRODUCT
template <class T, unsigned int blockSize>
__global__ void DOTGPU(T *c,
		const T *a,
		const T *b,
		const int Nx,
		const int Ny,
		const int Nz)
{
	extern __shared__ T cache[];

	unsigned int tid = threadIdx.x;
	unsigned int i = tid + blockIdx.x * (blockSize * 2);
	unsigned int gridSize = (blockSize*2)*gridDim.x;


	cache[tid] = 0;

	while(i<Nx*Ny*Nz) {
		cache[tid] += a[i+Nx*Ny] * b[i+Nx*Ny] + a[i+Nx*Ny+blockSize] * b[i+Nx*Ny+blockSize];
		i += gridSize;
	}

	__syncthreads();

	if(blockSize >= 512) {	if(tid < 256) { cache[tid] += cache[tid + 256]; } __syncthreads(); }
	if(blockSize >= 256) {	if(tid < 128) { cache[tid] += cache[tid + 128]; } __syncthreads(); }
	if(blockSize >= 128) {	if(tid < 64 ) { cache[tid] += cache[tid + 64 ]; } __syncthreads(); }

	if(tid < 32) {
		cache[tid] += cache[tid + 32];
		cache[tid] += cache[tid + 16];
		cache[tid] += cache[tid + 8];
		cache[tid] += cache[tid + 4];
		cache[tid] += cache[tid + 2];
		cache[tid] += cache[tid + 1];
	}

	if (tid == 0) c[blockIdx.x] = cache[0];
}


//
template <class T>
__global__ void elementWiseMul(T *x,
		const T *p,
		const int Nx,
		const int Ny)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x + Nx*Ny;
	x[tid] *= p[tid];
}


// Truncated Neumann series 2 in 3D
template <class T>
__global__ void makeTNS2(T *smC,
		T *smFF,
		T *smFS,
		T *smFW,
		T *smF,
		T *smFE,
		T *smFN,
		T *smSS,
		T *smSW,
		T *smS,
		T *smSE,
		T *smWW,
		T *smW,
		const T *stC,
		const T *stF,
		const T *stS,
		const T *stW,
		const int Nx,
		const int Ny,
		const int Nz,
		const int NxNyNz,
		const int NxNy)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int tids = tid + NxNy;  // tid shifted

	T C   = 1. / stC[tids];
	T S   = 0.;
	T W   = 0.;
	T F   = 0.;
	T WW  = 0.;
	T SW  = 0.;
	T SS  = 0.;
	T FW  = 0.;
	T FS  = 0.;
	T FF  = 0.;
	T SE  = 0.;
	T SSE = 0.;
	T FE  = 0.;
	T FN  = 0.;
	T FNW = 0.;
	T FFN = 0.;
	T FSE = 0.;

	if (tid < NxNyNz-NxNy-NxNy) 	FF	= 1. / stC[tids+NxNy+NxNy]; 	// FF
	if (tid < NxNyNz-NxNy-NxNy+Nx)  FFN	= 1. / stC[tids+NxNy+NxNy-Nx]; 	// FFN
	if (tid < NxNyNz-NxNy-Nx)   	FS	= 1. / stC[tids+NxNy+Nx];   	// FS
	if (tid < NxNyNz-NxNy-Nx+1)   	FSE	= 1. / stC[tids+NxNy+Nx-1];   	// FSE
	if (tid < NxNyNz-NxNy-1)   		FW	= 1. / stC[tids+NxNy+1];   		// FW
	if (tid < NxNyNz-NxNy)   		F	= 1. / stC[tids+NxNy];    		// F
	if (tid < NxNyNz-NxNy+Nx-1) 	FNW = 1. / stC[tids+NxNy-Nx+1]; 	// FNW
	if (tid < NxNyNz-NxNy+Nx)   	FN	= 1. / stC[tids+NxNy-Nx];   	// FN
	if (tid < NxNyNz-NxNy+1)    	FE  = 1. / stC[tids+NxNy-1];    	// FE
	if (tid < NxNyNz-2*Nx)   		SS  = 1. / stC[tids+2*Nx];      	// SS
	if (tid < NxNyNz-2*Nx+1)   		SSE = 1. / stC[tids+2*Nx-1];    	// SSE
	if (tid < NxNyNz-Nx-1)   		SW  = 1. / stC[tids+Nx+1];      	// SW
	if (tid < NxNyNz-Nx)     		S	= 1. / stC[tids+Nx];      		// S
	if (tid < NxNyNz-Nx+1)      	SE  = 1. / stC[tids+Nx-1];      	// SE
	if (tid < NxNyNz-2)      		WW	= 1. / stC[tids+2];				// WW
	if (tid < NxNyNz-1)      		W	= 1. / stC[tids+1];       		// W

    smC[tid+NxNy] = 1. * C 																						    	// C         OK
    		      + (stW[tid+1]    * C) * (stW[tid+1]       * C) 												  * W  		// W         OK
    		      + (stS[tid+Nx]   * C) * (stS[tid+Nx]      * C) 												  * S		// S         OK
    		      + (stF[tid+NxNy] * C) * (stF[tid+NxNy]    * C) 												  * F		// F         OK
    		      + (stW[tid+1]    * C) * (stW[tid+2]       * W) * (stW[tid+1]    * C) * (stW[tid+2]       * W)   * WW		// WW        OK
    		      +((stW[tid+1]    * C) * (stS[tid+Nx+1]    * W) + (stW[tid+Nx+1] * S) * (stS[tid+Nx]      * C))			// SW
    		      *((stW[tid+1]    * C) * (stS[tid+Nx+1]    * W) + (stW[tid+Nx+1] * S) * (stS[tid+Nx]      * C))  * SW      //		     OK
    		      + (stS[tid+Nx]   * C) * (stS[tid+2*Nx]    * S) * (stS[tid+Nx]   * C) * (stS[tid+2*Nx]    * S)   * SS		// SS        OK
    		      + (stW[tid+1]    * C) * (stF[tid+NxNy+1]  * W) * (stW[tid+1]    * C) * (stF[tid+NxNy+1]  * W)   * FW   	// FW        OK
    		      +((stS[tid+Nx]   * C) * (stF[tid+NxNy+Nx] * S) + (stF[tid+NxNy] * C) * (stS[tid+NxNy+Nx] * F))  			// FS
    		      *((stS[tid+Nx]   * C) * (stF[tid+NxNy+Nx] * S) + (stF[tid+NxNy] * C) * (stS[tid+NxNy+Nx] * F))  * FS      //			 OK
    		      + (stF[tid+NxNy] * C) * (stF[tid+2*NxNy]  * F) * (stF[tid+NxNy] * C) * (stF[tid+2*NxNy]  * F)   * FF;     // FF        OK

    smW[tid+1]     = -stW[tid+1]    * C  * W																										// OK
    		       + (stW[tid+1]    * C) * (stW[tid+2]       * W)                                                 * WW * (-stW[tid+2]       * W)    // OK
    		       +((stW[tid+1]    * C) * (stS[tid+Nx+1]    * W) + (stW[tid+Nx+1] * S) * (stS[tid+Nx]      * C)) * SW * (-stS[tid+Nx+1]    * W)    // OK
    		       + (stW[tid+1]    * C) * (stF[tid+NxNy+1]  * W)                                                 * FW * (-stF[tid+NxNy+1]  * W);   // OK

    smS[tid+Nx]    = -stS[tid+Nx]   * C  * S																										// OK
    		       +((stW[tid+1]    * C) * (stS[tid+Nx+1]    * W) + (stW[tid+Nx+1] * S) * (stS[tid+Nx]      * C)) * SW * (-stW[tid+Nx+1]    * S)    // OK
    		       + (stS[tid+Nx]   * C) * (stS[tid+2*Nx]    * S)                                                 * SS * (-stS[tid+2*Nx]    * S)    // OK
    		       +((stS[tid+Nx]   * C) * (stF[tid+NxNy+Nx] * S) + (stF[tid+NxNy] * C) * (stS[tid+NxNy+Nx] * F)) * FS * (-stF[tid+NxNy+Nx] * S);   // OK

    smF[tid+NxNy]  = -stF[tid+NxNy] * C  * F																										// OK
    		       + (stW[tid+1]    * C) * (stF[tid+NxNy+1]  * W)                                                 * FW * (-stW[tid+NxNy+1]  * F)	// OK
    		       +((stS[tid+Nx]   * C) * (stF[tid+NxNy+Nx] * S) + (stF[tid+NxNy] * C) * (stS[tid+NxNy+Nx] * F)) * FS * (-stS[tid+NxNy+Nx] * F)    // OK
    		       + (stF[tid+NxNy] * C) * (stF[tid+2*NxNy]  * F) 												  * FF * (-stF[tid+2*NxNy]  * F);   // OK

    smWW[tid+2]       = (stW[tid+1]    * C) * (stW[tid+2]      * W) * WW;													// ok   OK
    smSS[tid+2*Nx]    = (stS[tid+Nx]   * C) * (stS[tid+2*Nx]   * S) * SS;													// ok   OK
    smSW[tid+Nx+1]    =((stW[tid+1]    * C) * (stS[tid+Nx+1]   * W) + (stW[tid+Nx+1] * S) * (stS[tid+Nx]      * C)) * SW;	// ok   OK
    smFW[tid+NxNy+1]  = (stW[tid+1]    * C) * (stF[tid+NxNy+1] * W) * FW;                                                   // 		OK
    smFS[tid+NxNy+Nx] =((stS[tid+Nx]   * C) * (stF[tid+NxNy+Nx]* S) + (stF[tid+NxNy] * C) * (stS[tid+NxNy+Nx] * F)) * FS;   // 		OK
    smFF[tid+2*NxNy]  = (stF[tid+NxNy] * C) * (stF[tid+2*NxNy] * F) * FF;													// 		OK


    smSE[tid+Nx-1]   = (-stS[tid+Nx] * C)                                                                              * S  * (-stW[tid+Nx] * SE)							// fine
		             +(( stW[tid+1]  * C)  * (stS[tid+Nx+1]    * W) + (stW[tid+Nx+1] * S)   * (stS[tid+Nx]      * C))  * SW * ( stW[tid+Nx] * SE) * (stW[tid+Nx+1] * S)		// fine
		             + ( stS[tid+Nx] * C)  * (stS[tid+2*Nx]    * S)                                                    * SS
		             *(( stW[tid+Nx] * SE) * (stS[tid+2*Nx]    * S) + (stW[tid+2*Nx] * SSE) * (stS[tid+2*Nx-1]  * SE))														// fine
		             +(( stS[tid+Nx] * C)  * (stF[tid+NxNy+Nx] * S) + (stF[tid+NxNy] * C)   * (stS[tid+NxNy+Nx] * F))  * FS
		             * ( stW[tid+Nx] * SE) * (stF[tid+NxNy+Nx] * S);																										// fine

    smFE[tid+NxNy-1] = (-stF[tid+NxNy] * C * F) * (-stW[tid+NxNy] * FE)																// fine
		             + ( stW[tid+1]    * C)  * (stF[tid+NxNy+1]  * W)  * FW * (stW[tid+NxNy]    * FE) * (stW[tid+NxNy+1]  * F)		// fine
		             + ( stF[tid+NxNy] * C)  * (stF[tid+2*NxNy]  * F)  * FF * (stW[tid+NxNy]    * FE) * (stF[tid+2*NxNy]  * F)		// fine
		             +(( stS[tid+Nx]   * C)  * (stF[tid+NxNy+Nx] * S)  + (stF[tid+NxNy]    * C)   * (stS[tid+NxNy+Nx]   * F )) * FS
		             *(( stW[tid+NxNy] * FE) * (stS[tid+NxNy+Nx] * F)  + (stW[tid+NxNy+Nx] * FSE) * (stS[tid+NxNy+Nx-1] * FE));     // fine

    smFN[tid+NxNy-Nx] = (-stF[tid+NxNy]      * C)  * F * (-stS[tid+NxNy] * FN)														// fine
    		          + ( stW[tid+1]         * C)  * (stF[tid+NxNy+1]  * W) * FW
    		          *(( stW[tid+NxNy-Nx+1] * FN) * (stS[tid+NxNy+1]  * FNW) + (stW[tid+NxNy+1] * F) * (stS[tid+NxNy] * FN))	    // fine
    		          +((stS[tid+Nx]   * C)  * (stF[tid+NxNy+Nx] * S)  + (stF[tid+NxNy] * C) * (stS[tid+NxNy+Nx] * F)) * FS
    		          * (stS[tid+NxNy] * FN) * (stS[tid+NxNy+Nx] * F)
    		          + (stF[tid+NxNy] * C)  * (stF[tid+2*NxNy]  * F)  * FF
    		          *((stS[tid+NxNy] * FN) * (stF[tid+2*NxNy]  * F)  + (stF[tid+2*NxNy-Nx] * FN) * (stS[tid+2*NxNy] * FFN));		// fine
}


// for thermal boundary condition
template <class T>
__global__ void addNeumannBC(T *x,
		const T *Q,
		const T HeatFlux,
		const int Nx,
		const int Ny)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	x[tid+Nx*Ny] += HeatFlux * Q[tid];
}

#endif /* CUDAFUNCTIONS_H_ */
