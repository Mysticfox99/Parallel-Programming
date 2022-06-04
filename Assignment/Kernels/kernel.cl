//a very simple histogram implementation
kernel void hist_local_simple(global const uchar* A, global uint* H, local uint* LH, global uint* nr_bins, global uint* binmax)
{
	int id = get_global_id(0); 
	int lid = get_local_id(0);
	int bin_index = (A[id] / (double) binmax[0]) * (nr_bins[0] -1);
	LH[lid] = 0;
	barrier(CLK_LOCAL_MEM_FENCE);
	atomic_inc(&LH[bin_index]);
	barrier(CLK_LOCAL_MEM_FENCE);
	if (lid < nr_bins[0]) //combine all local hist into a global one
		atomic_add(&H[lid], LH[lid]);
}


//Hillis-Steele basic inclusive scan
//requires additional buffer B to avoid data overwrite 
kernel void scan_hs(global const uint* A, global uint* B) {
	int id = get_global_id(0);
	int N = get_global_size(0);
	global uint* C;
	for (int stride = 1; stride <= N; stride *= 2) 
	{
		B[id] = A[id];
		if (id >= stride)
		{
			B[id] += A[id - stride];
		}
		barrier(CLK_GLOBAL_MEM_FENCE); //sync the step
		C = A; A = B; B = C; //swap A & B between steps
	}
}

kernel void normalise(global const uint* A, global uint* B, global uint* binmax)
{
	int id = get_global_id(0);
	int N = get_global_size(0);
	int max = A[N - 1];
	B[id] = (A[id] / (double) max) * (binmax[0]-1);
}

kernel void back(global const uchar * A, global uint * B, global uchar* C, global uint* nr_bins, global uint* binmax)
{
	int id = get_global_id(0);
	int bin_index = (A[id] / (double)binmax[0]) * (nr_bins[0] - 1);
	C[id] = B[bin_index];
}
