#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>

#include "adios2.h"
#include "mgard/mgard_api.h"

#define SECONDS(d) std::chrono::duration_cast<std::chrono::seconds>(d).count()

template <typename Type>
void FileWriter_bin(const char *filename, Type *data, size_t size)
{
  std::ofstream fout(filename, std::ios::binary);
  fout.write((const char*)(data), size*sizeof(Type));
  fout.close();
}

int main(int argc, char **argv) {
	std::chrono::steady_clock clock;
	std::chrono::steady_clock::time_point start, stop;
	std::chrono::steady_clock::duration duration;

    float tol, result;
    int out_size;
    unsigned char *compressed_data = 0;

    MPI_Init(&argc, &argv);
    int rank, np_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np_size);

    std::vector<float> rel_tol {1e-3, 1e-3, 1e-3, 1e-4, 1e-4};
    adios2::ADIOS ad;
    std::string dpath("/gpfs/alpine/proj-shared/csc143/gongq/andes/TC-CaseStudy/mgard-test/stb_layout/");
    std::string fname(argv[1]);
    std::vector<std::string> data_f{"01-01", "01-31", "03-02", "04-01", "05-01", "05-31", "06-30", "07-30", "08-29", "09-28", "10-28", "11-27", "12-27"};
    std::vector<std::string> var_name {"PSL", "T200", "T500", "UBOT", "VBOT"};
    std::vector<std::string> suffx = {"_top.bp", "_bottom.bp", "_side.bp"};
    size_t data_id = (size_t)std::floor((double)rank/suffx.size());
    size_t suff_id = (size_t)std::fmod((double)rank, suffx.size());
    while (data_id < data_f.size()) {
        std::vector<std::size_t> compressed_size(5,0);
        std::cout << "rank: " << rank << ", data: " << data_f[data_id] << "\n";
        adios2::IO reader_io = ad.DeclareIO("Input"+std::to_string(data_id)+std::to_string(suff_id));
        adios2::IO writer_io = ad.DeclareIO("Output"+std::to_string(data_id)+std::to_string(suff_id));
        adios2::Engine reader = reader_io.Open(dpath+fname+data_f[data_id]+"-21600.tcv5"+suffx[suff_id], adios2::Mode::Read);
        adios2::Engine writer = writer_io.Open("./2D/1e-4-gb/"+fname+data_f[data_id]+"-21600.tcv5" +suffx[suff_id] + ".mgard", adios2::Mode::Write);
        for (int ivar=0; ivar<var_name.size(); ivar++) { // MPI ADIOS: decompose the variable  
            adios2::Variable<float> var_ad2;
            var_ad2 = reader_io.InquireVariable<float>(var_name[ivar]);
            std::vector<std::size_t> shape = var_ad2.Shape();
            var_ad2.SetSelection(adios2::Box<adios2::Dims>({0, 0, 0}, shape));
            std::cout << "rank " << rank << " read " << var_name[ivar] << " in " << suffx[suff_id] << " from {0, 0, 0} to {" << shape[0] << ", " << shape[1] << ", " << shape[2] << "}\n";
            std::vector<float> var_in;
            reader.Get<float>(var_ad2, var_in, adios2::Mode::Sync);
            reader.PerformGets();

            float *dcp_var = new float[shape[0]*shape[1]*shape[2]]; 
            const std::array<std::size_t, 2> dims = {shape[1], shape[2]};
            const mgard::TensorMeshHierarchy<2, float> hierarchy(dims);
            const size_t ndof = hierarchy.ndof();
            auto [min_v, max_v] = std::minmax_element(begin(var_in), end(var_in));
            tol = rel_tol.at(ivar) * (*max_v- *min_v);
            std::cout << "tol: " << tol << "\n";
            for (size_t it=0; it<shape[0]; it++) {
                std::cout << "step " << it << "\n";//<< ": maxv = " << *max_v << ", min_v = " << *min_v << ", abs eb: " << tol << "\n";
                const mgard::CompressedDataset<2, float> compressed = mgard::compress(hierarchy, var_in.data()+it*shape[1]*shape[2], (float)0.0, tol);
                const mgard::DecompressedDataset<2, float> decompressed = mgard::decompress(compressed);
                memcpy(&dcp_var[it*shape[1]*shape[2]], decompressed.data(), shape[1]*shape[2]*sizeof(float));
                compressed_size[ivar] += compressed.size();
            }
            adios2::Variable<float>var_out = writer_io.DefineVariable<float>(var_name[ivar], shape, {0, 0, 0}, shape, adios2::ConstantDims);
            writer.Put<float>(var_out, dcp_var, adios2::Mode::Sync);
            writer.PerformPuts();
            delete dcp_var;
        }
        writer.Close();
        reader.Close();
        std::cout << "processor " << rank << ", " << var_name[0] << ": " << compressed_size[0] << ", " << var_name[1] << ": " << compressed_size[1] << ", " << var_name[2] << ": " << compressed_size[2] << ", " << var_name[3] << ": " << compressed_size[3] << ", " << var_name[4] << ": " << compressed_size[4] << "\n"; 
        rank += np_size;
        data_id = (int)std::floor((double)rank/suffx.size());
        suff_id = (size_t)std::fmod((double)rank, suffx.size());
//        std::cout << data_id << ", " << suff_id << "\n";
    }
    return(0);
}
