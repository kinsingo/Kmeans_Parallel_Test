#include "CL_Compute.h"

CL_Compute::CL_Compute(string kernel_cl_file_path)
{
    context = cl::Context(Get_default_device());
    program = Get_Program_From_kernel_cl_file_path(kernel_cl_file_path);
}

cl::Context CL_Compute::Get_Context() { return context; }
cl::Program CL_Compute::Get_Program() { return program; }


void CL_Compute::Print_Platform_Info()
{
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    for (const auto& platform : platforms)
    {
        std::string platformName;
        platform.getInfo(CL_PLATFORM_NAME, &platformName);

        std::string platformVersion;
        platform.getInfo(CL_PLATFORM_VERSION, &platformVersion);

        std::cout << "Platform Name: " << platformName << std::endl;
        std::cout << "Platform Version: " << platformVersion << std::endl;
    }
}

cl::Device CL_Compute::Get_default_device()
{
    //get all platforms (drivers)
    vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);
    if (all_platforms.size() == 0) {
        std::cout << " No platforms found. Check OpenCL installation!\n";
        exit(1);
    }

    for (cl::Platform platform : all_platforms)
        std::cout << " available platform : " << platform.getInfo<CL_PLATFORM_NAME>() << endl;;


    cl::Platform default_platform = all_platforms[0];
    std::cout << "Using platform(default_platform): " << default_platform.getInfo<CL_PLATFORM_NAME>() << "\n";

    //get default device of the default platform
    std::vector<cl::Device> all_devices;
    default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
    if (all_devices.size() == 0) {
        cout << " No devices found. Check OpenCL installation!\n";
        exit(1);
    }
    cl::Device default_device = all_devices[0];
    cout << "Using device: " << default_device.getInfo<CL_DEVICE_NAME>() << "\n";

    return default_device;
}

cl::Program::Sources CL_Compute::Get_Sources_From_kernel_cl_file_path(string kernel_cl_file_path)
{
    std::ifstream file(kernel_cl_file_path);
    if (file.is_open() == false)
        throw exception("Failed to open kernel file.");

    std::string kernel_String(std::istreambuf_iterator<char>(file), (std::istreambuf_iterator<char>()));
    file.close();
    cout << kernel_String << endl;

    cl::Program::Sources sources;
    const char* s = kernel_String.c_str();
    sources.push_back({ s,strlen(s) });
    return sources;
}

cl::Program CL_Compute::Get_Program_From_kernel_cl_file_path(string kernel_cl_file_path)
{
    cl::Program program(context, Get_Sources_From_kernel_cl_file_path(kernel_cl_file_path));
    program.build();
    return program;
}

cl::Buffer CL_Compute::Get_ReadOnlyBuffer(cl::Context context, Mat mat)
{
    return cl::Buffer(context, sizeof(float) * mat.total(), true);//readonly
}

cl::Buffer CL_Compute::Get_Buffer(cl::Context context, Mat mat)
{
    return cl::Buffer(context, sizeof(float) * mat.total(), false);//write and read
}