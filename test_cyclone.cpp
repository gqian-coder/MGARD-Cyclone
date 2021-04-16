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

template<typename Type>
void FileWriter_ad(const char *filename, Type *data, std::vector<size_t> size)
{
  adios2::ADIOS ad;
  adios2::IO bpIO = ad.DeclareIO("WriteBP_File");
  adios2::Variable<Type> bp_fdata = bpIO.DefineVariable<Type>(
        "i_f", size, {0,0,0,0}, size,  adios2::ConstantDims);
  // Engine derived class, spawned to start IO operations //
  adios2::Engine bpFileWriter = bpIO.Open(filename, adios2::Mode::Write);
  bpFileWriter.Put<Type>(bp_fdata, data);
  bpFileWriter.Close();
}


int main(int argc, char **argv) {
	std::chrono::steady_clock clock;
	std::chrono::steady_clock::time_point start, stop;
	std::chrono::steady_clock::duration duration;

    float tol, rel_tol, result;
    int out_size;
    unsigned char *compressed_data = 0;

    MPI_Init(&argc, &argv);
    int rank, comm_size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    rel_tol = 1e-4;
    adios2::ADIOS ad;
    adios2::IO reader_io  = ad.DeclareIO("XGC");
    std::string dpath("/gpfs/alpine/proj-shared/csc143/gongq/andes/TC-CaseStudy/mgard-test/");
    std::string fname(argv[1]);
    std::vector<std::string> data_f{"01-01", "01-31", "03-02", "04-01", "05-01", "05-31", "06-30", "07-30", "08-29", "09-28", "10-28", "11-27", "12-27"};
    adios2::Engine reader = reader_io.Open(dpath+fname+data_f[rank]+"-21600.tcv5.bp", adios2::Mode::Read);
    std::vector<std::string> var_name {"PSL", "T200", "T500", "UBOT", "VBOT"};

    adios2::IO writer_io = ad.DeclareIO("Output");
    adios2::Engine writer = writer_io.Open(fname+data_f[rank]+"-21600.tcv5.bp"+".mgard", adios2::Mode::Write);
    std::vector<size_t> compressed_size (5,0);
    for (int ivar=0; ivar<var_name.size(); ivar++) {
        std::cout << "compress " << var_name[ivar] << "\n";
        // Inquire variable
        adios2::Variable<float> var_ad2;
        var_ad2 = reader_io.InquireVariable<float>(var_name[ivar]);
        std::vector<std::size_t> shape = var_ad2.Shape();
        var_ad2.SetSelection(adios2::Box<adios2::Dims>({0, 0}, shape));
        std::cout << shape[0] << " " << shape[1] << "\n";
        std::vector<float> var_in;
        reader.Get<float>(var_ad2, var_in, adios2::Mode::Sync);
    
        float *dcp_var = new float[shape[0]*shape[1]]; 
        const std::array<std::size_t, 1> dims = {shape[0]};
        const mgard::TensorMeshHierarchy<1, float> hierarchy(dims);
        const size_t ndof = hierarchy.ndof();
        for (size_t it=0; it<shape[1]; it++) {
            auto [min_v, max_v] = std::minmax_element(begin(var_in)+it*shape[0], begin(var_in)+(it+1)*shape[0]);
            tol = rel_tol * (*max_v- *min_v);
            std::cout << "step " << it << "\n";//<< ": maxv = " << *max_v << ", min_v = " << *min_v << ", abs eb: " << tol << "\n";
            const mgard::CompressedDataset<1, float> compressed = mgard::compress(hierarchy, var_in.data()+it*shape[0], (float)0.0, tol);
            const mgard::DecompressedDataset<1, float> decompressed = mgard::decompress(compressed);
            memcpy(&dcp_var[it*shape[0]], decompressed.data(), shape[0]*sizeof(float));
            compressed_size[ivar] += compressed.size();
        }
        adios2::Variable<float>var_out = writer_io.DefineVariable<float>(var_name[ivar], shape, {0, 0}, shape, adios2::ConstantDims);
        writer.Put<float>(var_out, dcp_var, adios2::Mode::Sync);
        delete dcp_var;
    }
    writer.Close();
    reader.Close();
    std::cout << "processor " << rank << ": after compression " << var_name[0] << " cp_sz=" << compressed_size[0] << ", " <<  var_name[1] << " cp_sz=" << compressed_size[1] << ", " << var_name[2] << " cp_sz=" << compressed_size[2] << ", " << var_name[3] << " cp_sz=" << compressed_size[3] << ", " << var_name[4] << " cp_sz=" << compressed_size[4] << "\n";
    return(0);
}
