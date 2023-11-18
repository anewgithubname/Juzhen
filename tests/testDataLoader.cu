#include "../ml/dataloader.hpp"
#include <fstream>

int test1()
{
    using namespace Juzhen;
    
    std::string base = PROJECT_DIR;

    int n_total = 10000;

    DataLoader<float, int> loader(base + "/datasets/MNIST", "test", 34);
    //create folder images
    std::string command = "mkdir -p \"" + base + "/datasets/MNIST/images\"";
    system(command.c_str());
    
    for(int i = 0; i < 10000; i ++){
        auto [x, y] = loader.next_batch();
        // write the batch as a grayscale pbm image
        for(int img = 0; img < x.num_col(); img ++){
            int label = y(0, img);
            //name the image end with its label
            std::string imagename = base + "/datasets/MNIST/images/" + std::to_string(i*34 + img) 
                                         + "_" + std::to_string(label) + ".pbm";
            std::ofstream ofs(imagename, std::ios::out | std::ios::binary);
            ofs << "P2\n28 28\n255\n";
            for(int row = 0; row < 28; row ++){
                for(int col = 0; col < 28; col ++){
                    ofs << x(row*28 + col, img) << " ";
                }
                ofs << "\n";
            }
            ofs.close();
        }
        
        // check if we have insufficient number of samples
        if(x.num_col() < 34){
            if(i*34 + x.num_col() == n_total){
                return 0; 
            }else{
                return 1;
            }
        }

    }
    return 0;
}


int compute()
{
    spdlog::set_level(spdlog::level::debug);
    std::cout << __cplusplus << " " << HAS_CONCEPTS << std::endl;

    int ret = 0;
    ret += test1();
    std::cout << std::endl;

    //zip the images folder
    std::string base = PROJECT_DIR;
    std::string command = "cd " + base + "/datasets/MNIST/images && zip -r ../images.zip .";
    system(command.c_str());

//check if we are on windows
#ifdef _WIN64
    //remove the images folder
    command = "rmdir /s /q \"" + base + "/datasets/MNIST/images\"";
    system(command.c_str());
#else
    //remove the images folder
    command = "rm -rf " + base + "/datasets/MNIST/images";
    system(command.c_str());
#endif

    if (ret == 0)
    {
        LOG_INFO("--------------------");
        LOG_INFO("|      ALL OK!     |");
        LOG_INFO("--------------------");
    }
    else
    {
        LOG_ERROR("--------------------");
        LOG_ERROR("|    NOT ALL OK!   |");
        LOG_ERROR("--------------------");
    }
        
    return ret;
}