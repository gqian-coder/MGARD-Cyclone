#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>

#include "adios2.h"
#include "mgard/mgard_api.h"

#define SECONDS(d) std::chrono::duration_cast<std::chrono::seconds>(d).count()

#define T_STEP 4

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

    std::vector<float> rel_tol {1e-4, 1e-4, 1e-4, 1e-4, 1e-4};
    adios2::ADIOS ad(MPI_COMM_WORLD);
    std::string dpath("/gpfs/alpine/proj-shared/csc143/gongq/andes/TC-CaseStudy/mgard-test/stb_layout/");
    std::string fname(argv[1]);
    std::vector<std::string> data_f{"01-01"/*, "01-31", "03-02", "04-01", "05-01", "05-31", "06-30", "07-30", "08-29", "09-28", "10-28", "11-27", "12-27"*/};
    std::vector<std::string> var_name {"PSL", "T200", "T500", "UBOT", "VBOT"};
    std::vector<std::string> suffx = {"_top.bp", "_bottom.bp", "_side.bp"};
    std::vector<std::size_t> compressed_size(var_name.size(),0);
    for (size_t data_id=0; data_id<data_f.size(); data_id++) {
        for (size_t suff_id=0; suff_id<suffx.size(); suff_id++) {
            if (rank==0)
                std::cout << "readin: " << data_f[data_id] << ": " << suffx[suff_id] << "\n";
            adios2::IO reader_io = ad.DeclareIO("Input"+std::to_string(data_id)+std::to_string(suff_id));
            adios2::IO writer_io = ad.DeclareIO("Output"+std::to_string(data_id)+std::to_string(suff_id));
            adios2::Engine reader = reader_io.Open(dpath+fname+data_f[data_id]+"-21600.tcv5"+suffx[suff_id], adios2::Mode::Read);
            adios2::Engine writer = writer_io.Open("./3D/1e-4/step4/"+fname+data_f[data_id]+"-21600.tcv5" +suffx[suff_id] + ".mgard", adios2::Mode::Write);
            for (int ivar=0; ivar<var_name.size(); ivar++) { // MPI ADIOS: decompose the variable  
                size_t r_step = rank*T_STEP;
                adios2::Variable<float> var_ad2;
                var_ad2 = reader_io.InquireVariable<float>(var_name[ivar]);
                std::vector<std::size_t> shape = var_ad2.Shape();
                adios2::Variable<float>var_out = writer_io.DefineVariable<float>(var_name[ivar], shape, {0, 0, 0}, shape);
                const std::array<std::size_t, 3> dims = {4, shape[1], shape[2]};
                const mgard::TensorMeshHierarchy<3, float> hierarchy(dims);
                const size_t ndof = hierarchy.ndof();
                while (r_step<shape[0]) {
                    std::vector<float> var_in;
                    var_ad2.SetSelection(adios2::Box<adios2::Dims>({r_step, 0, 0}, {4, shape[1], shape[2]}));
                    reader.Get<float>(var_ad2, var_in, adios2::Mode::Sync);
                    reader.PerformGets();
                    auto [min_v, max_v] = std::minmax_element(begin(var_in), end(var_in));
                    tol = rel_tol.at(ivar) * (*max_v- *min_v);
//                    std::cout << "tol: " << tol << "\n";
//                    std::cout << "rank " << rank << " read " << var_name[ivar] << " in " << suffx[suff_id] << " step " << r_step << "\n";
                    const mgard::CompressedDataset<3, float> compressed = mgard::compress(hierarchy, var_in.data(), (float)0.0, tol);
                    const mgard::DecompressedDataset<3, float> decompressed = mgard::decompress(compressed);
                    compressed_size[ivar] += compressed.size();
                    var_out.SetSelection(adios2::Box<adios2::Dims>({r_step, 0, 0}, {4, shape[1], shape[2]})); 
                    writer.Put<float>(var_out, decompressed.data(), adios2::Mode::Sync);
                    writer.PerformPuts();
                    r_step += np_size*T_STEP; 
                }
            }
            writer.Close();
            reader.Close();
        }
    }
    std::cout << "processor " << rank << ", " << var_name[0] << ": " << compressed_size[0] << ", " << var_name[1] << ": " << compressed_size[1] << ", " << var_name[2] << ": " << compressed_size[2] << ", " << var_name[3] << ": " << compressed_size[3] << ", " << var_name[4] << ": " << compressed_size[4] << "\n";
    MPI_Finalize(); 
    return 0;
}
